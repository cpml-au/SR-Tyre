import argparse
from flex.gp.util import (
    detect_nested_trigonometric_functions,
    compile_individual_with_consts,
    load_config_data,
)
from flex.gp import regressor as gps
from flex.gp.primitives import add_primitives_to_pset_from_dict
import numpy as np
import warnings
import pygmo as pg
import mygrad as mg
from mygrad._utils.lock_management import mem_guard_off
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from deap import gp
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import ray

from ..force import compute_force_model
from ..plot import plot_force_model_data
from ..process_data import flatten_bins, load_and_process_bins, load_and_process_dataset

# set up number of cpus per ray worker
num_cpus = 1
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().with_name("simple_sr.yaml")
RESULTS_PATH = ROOT_DIR / "models" / "best_model_results.txt"
MODEL_THETA_OPT = np.array([
    0.0668, 0.0001, 360.4850, 0.0230,
    0.6456, 3.07e-05,
    1.7213, 3.6888, 1.3652,
])
MODEL_V = 16
MODEL_N_V = 200
MODEL_N_X = 100


# --- Custom generate dataset function ---
def generate_dataset(scaleXy: bool = True):
    np.random.seed(42)
    num_variables = 1
    scaler_X = None
    scaler_y = None

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_and_process_dataset()
    x_bins, y_bins, Fz_rep = load_and_process_bins()
    split_data = {
        "train": {
            "x_bins": x_bins[:3],
            "y": flatten_bins(y_bins[:3]),
            "Fz_rep": Fz_rep[:3],
        },
        "val": {
            "x_bins": x_bins[3:4],
            "y": flatten_bins(y_bins[3:4]),
            "Fz_rep": Fz_rep[3:4],
        },
        "test": {
            "x_bins": x_bins[4:5],
            "y": flatten_bins(y_bins[4:5]),
            "Fz_rep": Fz_rep[4:5],
        },
    }

    X_train = X_train.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    num_variables = X_train.shape[1]

    if scaleXy:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        X_test_scaled = scaler_X.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        y_train_scaled = y_train
        y_val_scaled = y_val
        X_test_scaled = X_test

    y_val_scaled = y_val_scaled.flatten()
    y_test = y_test.flatten()
    y_train_scaled = y_train_scaled.flatten()

    num_train_points = X_train.shape[0]

    # note y_test and y_train_scaled must be flattened
    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test,
        scaler_X,
        scaler_y,
        num_variables,
        num_train_points,
        split_data,
    )


def inverse_transform_targets(y, scaler_y):
    y = np.asarray(y).reshape(-1, 1)
    if scaler_y is None:
        return y.flatten()
    return scaler_y.inverse_transform(y).flatten()


def compute_regression_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def predict_force_model_from_callable(individual, x_bins, Fz_rep, consts=[]):
    def mu_expression(v, mu_s, v_s, delta_s):
        v_input = np.asarray(v).reshape(-1, 1)
        mu = eval_model(individual, v_input, consts)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        return np.nan_to_num(mu, nan=1.0, posinf=1e8, neginf=-1e8)

    v_model, F_b = compute_force_model(
        Fz_rep,
        MODEL_THETA_OPT,
        V=MODEL_V,
        n_v=MODEL_N_V,
        n_x=MODEL_N_X,
        mu_expression=mu_expression,
    )

    y_pred_bins = []
    for k, x_bin in enumerate(x_bins):
        if len(x_bin) == 0:
            continue
        v_query = -x_bin * MODEL_V
        y_model = F_b[k] / Fz_rep[k]
        y_pred_bins.append(np.interp(v_query, v_model, y_model))

    return flatten_bins(y_pred_bins)


def predict_force_model_with_regressor(gpsr, x_bins, Fz_rep):
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()
    individual, _ = compile_individual_with_consts(gpsr._best, toolbox)
    consts = getattr(gpsr._best, "consts", [])
    return predict_force_model_from_callable(individual, x_bins, Fz_rep, consts=consts)


def make_mu_expression_from_regressor(gpsr):
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()
    individual, _ = compile_individual_with_consts(gpsr._best, toolbox)
    consts = getattr(gpsr._best, "consts", [])

    def mu_expression(v, mu_s, v_s, delta_s):
        v_input = np.asarray(v).reshape(-1, 1)
        mu = eval_model(individual, v_input, consts)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        return np.nan_to_num(mu, nan=1.0, posinf=1e8, neginf=-1e8)

    return mu_expression


def save_model_results(
    gpsr,
    validation_score,
    params,
    split_data,
):
    y_train_pred = predict_force_model_with_regressor(
        gpsr, split_data["train"]["x_bins"], split_data["train"]["Fz_rep"]
    )
    y_val_pred = predict_force_model_with_regressor(
        gpsr, split_data["val"]["x_bins"], split_data["val"]["Fz_rep"]
    )
    y_test_pred = predict_force_model_with_regressor(
        gpsr, split_data["test"]["x_bins"], split_data["test"]["Fz_rep"]
    )

    y_train_true = np.asarray(split_data["train"]["y"])
    y_val_true = np.asarray(split_data["val"]["y"])
    y_test_true = np.asarray(split_data["test"]["y"])

    train_rmse, train_r2 = compute_regression_metrics(y_train_true, y_train_pred)
    val_rmse, val_r2 = compute_regression_metrics(y_val_true, y_val_pred)
    test_rmse, test_r2 = compute_regression_metrics(y_test_true, y_test_pred)

    result_lines = [
        f"best_expression: {gpsr._best}",
        f"best_validation_score: {validation_score}",
        f"best_hyperparameters: {params}",
        f"train_rmse: {train_rmse}",
        f"train_r2: {train_r2}",
        f"val_rmse: {val_rmse}",
        f"val_r2: {val_r2}",
        f"test_rmse: {test_rmse}",
        f"test_r2: {test_r2}",
    ]
    RESULTS_PATH.write_text("\n".join(result_lines) + "\n")


def print_dataset_info(
    X_train_scaled,
    y_train_scaled,
    X_val_scaled,
    y_val_scaled,
    X_test_scaled,
    y_test,
):
    print("Dataset summary:")
    print(f"  Train: X shape={X_train_scaled.shape}, y shape={y_train_scaled.shape}")
    print(f"  Validation: X shape={X_val_scaled.shape}, y shape={y_val_scaled.shape}")
    print(f"  Test: X shape={X_test_scaled.shape}, y shape={y_test.shape}")


def eval_model(individual, X, consts=[]):
    num_variables = X.shape[1]
    if num_variables > 1:
        X = [X[:, i] for i in range(num_variables)]
    else:
        X = [X]
    warnings.filterwarnings("ignore")
    y_pred = individual(*X, consts)
    return y_pred


def compute_MSE(individual, X, y, consts=[]):
    y_pred = eval_model(individual, X, consts)
    MSE = np.mean((y - y_pred) ** 2)

    if np.isnan(MSE) or np.isinf(MSE):
        MSE = 1e8

    return MSE


def compute_force_model_MSE(individual, x_bins, y, Fz_rep, consts=[]):
    y_pred = predict_force_model_from_callable(
        individual,
        x_bins,
        Fz_rep,
        consts=consts,
    )
    MSE = np.mean((y - y_pred) ** 2)

    if np.isnan(MSE) or np.isinf(MSE):
        MSE = 1e8

    return MSE


def eval_MSE_and_tune_constants(tree, toolbox, x_bins, y, Fz_rep):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)

    if num_consts > 0:
        eval_MSE = partial(
            compute_force_model_MSE,
            individual=individual,
            x_bins=x_bins,
            y=y,
            Fz_rep=Fz_rep,
        )

        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err = eval_MSE(consts=x)
                # return [total_err + 0.*(np.linalg.norm(x, 2))**2]
                return [total_err]

            def gradient(self, x):
                with mem_guard_off:
                    xt = mg.tensor(x, copy=False)
                    f = self.fitness(xt)[0]
                    f.backward()
                return xt.grad

            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.nlopt(solver="lbfgs"))
        # algo = pg.algorithm(pg.pso(gen=10))
        # pop = pg.population(prb, size=70)
        algo.extract(pg.nlopt).maxeval = 10
        pop = pg.population(prb, size=1)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        MSE = pop.champion_f[0]
        consts = pop.champion_x

        if np.isinf(MSE) or np.isnan(MSE):
            MSE = 1e8
    else:
        MSE = compute_force_model_MSE(individual, x_bins, y, Fz_rep)
        consts = []
    return MSE, consts


def get_features_batch(individuals_batch, individ_feature_extractors=None):
    if individ_feature_extractors is None:
        individ_feature_extractors = [
            len,
            lambda ind: detect_nested_trigonometric_functions(str(ind)),
        ]

    features_batch = [
        [fe(i) for i in individuals_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    return individ_length, nested_trigs


def predict(individuals_batch, toolbox, X, penalty, fitness_scale):

    predictions = [None] * len(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        predictions[i] = eval_model(callable, X, consts=tree.consts)

    return predictions


def compute_MSEs(individuals_batch, toolbox, X, y, penalty, fitness_scale):

    total_errs = [None] * len(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        total_errs[i] = compute_MSE(callable, X, y, consts=tree.consts)

    return total_errs


def compute_attributes(
    individuals_batch,
    toolbox,
    X,
    y,
    penalty,
    fitness_scale,
    train_x_bins,
    train_y,
    train_Fz_rep,
):

    attributes = [None] * len(individuals_batch)

    individ_length, nested_trigs = get_features_batch(individuals_batch)

    for i, tree in enumerate(individuals_batch):

        # Tarpeian selection
        if individ_length[i] >= 50:
            consts = None
            fitness = (1e8,)
        else:
            MSE, consts = eval_MSE_and_tune_constants(
                tree,
                toolbox,
                train_x_bins,
                train_y,
                train_Fz_rep,
            )
            fitness = (
                fitness_scale
                * (
                    MSE
                    + 100000 * nested_trigs[i]
                    + penalty["reg_param"] * individ_length[i]
                ),
            )
        attributes[i] = {"consts": consts, "fitness": fitness}
    return attributes


def assign_attributes(individuals_batch, attributes):
    for ind, attr in zip(individuals_batch, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def score(individuals_batch, toolbox, X, y, penalty, fitness_scale):
    predictions = predict(individuals_batch, toolbox, X, penalty, fitness_scale)
    scores = [None] * len(individuals_batch)
    for i, prediction in enumerate(predictions):
        scores[i] = r2_score(y, prediction)
    return scores


def build_regressor(num_variables, params, cfgfile):
    regressor_params, config = load_config_data(cfgfile)
    regressor_params["num_individuals"] = params["num_individuals"]
    regressor_params["num_islands"] = params["num_islands"]

    batch_size = config["gp"]["batch_size"]
    penalty = config["gp"]["penalty"]
    fitness_scale = 1.0
    common_params = {
        "penalty": penalty,
        "fitness_scale": fitness_scale,
        "train_x_bins": params["split_data"]["train"]["x_bins"],
        "train_y": params["split_data"]["train"]["y"],
        "train_Fz_rep": params["split_data"]["train"]["Fz_rep"],
    }

    initial_individuals = config["gp"].get("initial_individuals", None)
    if not initial_individuals:
        initial_individuals = None

    pset = gp.PrimitiveSetTyped("Main", [float] * num_variables, float)
    pset = add_primitives_to_pset_from_dict(pset, config["gp"]["primitives"])
    if config["gp"]["use_constants"]:
        pset.addTerminal(object, float, "c")

    return gps.GPSymbolicRegressor(
        pset_config=pset,
        fitness=compute_attributes,
        predict_func=predict,
        score_func=score,
        common_data=common_params,
        callback_func=assign_attributes,
        num_best_inds_str=1,
        save_best_individual=True,
        save_train_fit_history=True,
        seed_str=initial_individuals,
        batch_size=batch_size,
        num_cpus=num_cpus,
        print_log=True,
        **regressor_params,
    )


# Custom optimization routine for Optuna.
# Although OptunaSearchCV could be used for this example,
# in practice one often needs a custom objective function.
# This shows how to define one and integrate it with Flex.
def optimize(trial, X, y, X_val, y_val, split_data, grid_search_parameters, cfgfile):
    num_variables = X.shape[1]
    params = {}
    for key in grid_search_parameters.keys():
        params[key] = trial.suggest_categorical(key, grid_search_parameters[key])

    gpsr = build_regressor(
        num_variables,
        params | {"split_data": split_data},
        cfgfile,
    )
    gpsr.fit(X, y)
    validation_predictions = predict_force_model_with_regressor(
        gpsr,
        split_data["val"]["x_bins"],
        split_data["val"]["Fz_rep"],
    )
    validation_score = r2_score(split_data["val"]["y"], validation_predictions)
    return validation_score


def main():
    if not ray.is_initialized():
        ray.init(runtime_env={"working_dir": str(ROOT_DIR)})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Enable Optuna hyperparameter optimization.",
    )
    args = parser.parse_args()

    grid_search_parameters = {
        "num_individuals": [250],
        "num_islands": [1],
    }

    regressor_params, config_file_data = load_config_data(str(CONFIG_PATH))

    scaleXy = config_file_data["gp"]["scaleXy"]

    # generate training and test datasets
    (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test,
        _,
        scaler_y,
        num_variables,
        _,
        split_data,
    ) = generate_dataset(scaleXy=scaleXy)

    print_dataset_info(
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test,
    )

    if args.hpo:
        sampler = TPESampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            partial(
                optimize,
                X=X_train_scaled,
                y=y_train_scaled,
                X_val=X_val_scaled,
                y_val=y_val_scaled,
                split_data=split_data,
                grid_search_parameters=grid_search_parameters,
                cfgfile=str(CONFIG_PATH),
            ),
            n_trials=max(
                1, int(np.prod([len(values) for values in grid_search_parameters.values()]))
            ),
        )
        best_params = study.best_trial.params
        best_validation_score = study.best_trial.value
    else:
        best_params = {
            "num_individuals": regressor_params["num_individuals"],
            "num_islands": regressor_params["num_islands"],
            "split_data": split_data,
        }
        best_gpsr = build_regressor(num_variables, best_params, str(CONFIG_PATH))
        best_gpsr.fit(X_train_scaled, y_train_scaled)
        best_validation_predictions = predict_force_model_with_regressor(
            best_gpsr,
            split_data["val"]["x_bins"],
            split_data["val"]["Fz_rep"],
        )
        best_validation_score = r2_score(
            split_data["val"]["y"],
            best_validation_predictions,
        )

    if args.hpo:
        best_gpsr = build_regressor(
            num_variables,
            best_params | {"split_data": split_data},
            str(CONFIG_PATH),
        )
        best_gpsr.fit(X_train_scaled, y_train_scaled)

    save_model_results(
        best_gpsr,
        best_validation_score,
        {k: v for k, v in best_params.items() if k != "split_data"},
        split_data,
    )

    x_bins, y_bins, Fz_rep = load_and_process_bins()
    plot_force_model_data(
        x_bins,
        y_bins,
        Fz_rep,
        theta_opt=MODEL_THETA_OPT,
        V=MODEL_V,
        n_v=MODEL_N_V,
        n_x=MODEL_N_X,
        mu_expression=make_mu_expression_from_regressor(best_gpsr),
    )

    print("Accuracy: {}".format(best_validation_score))
    print(
        "Best hyperparameters: {}".format(
            {k: v for k, v in best_params.items() if k != "split_data"}
        )
    )
    print("Best model results saved to {}".format(RESULTS_PATH))


if __name__ == "__main__":
    main()
