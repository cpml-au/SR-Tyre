import argparse
from flex.gp.util import (
    compile_individual_with_consts,
    load_config_data,
)
from flex.gp import regressor as gps
from flex.gp.primitives import add_primitives_to_pset_from_dict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from deap import gp
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import ray

from ..plot import plot_force_model_data
from ..process_data import load_and_process_dataset, load_split_representative_loads
from .fitness import (
    assign_attributes,
    compute_attributes,
    eval_model,
    predict,
    predict_force_model_with_regressor,
    score,
)
from .save_results import RESULTS_PATH, save_model_results
from functools import partial

# set up number of cpus per ray worker
num_cpus = 1
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
MODEL_V = 16
MODEL_N_V = 200
MODEL_N_X = 100
PLOT_PATH = Path(__file__).resolve().with_name("best_model_plot.png")


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
    (
        Fz_train_rep,
        Fz_val_rep,
        Fz_test_rep,
        Fz_overall_rep,
    ) = load_split_representative_loads()

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

        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        X_test = scaler_X.transform(X_test)

    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    num_train_points = X_train.shape[0]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler_X,
        scaler_y,
        num_variables,
        num_train_points,
        Fz_train_rep,
        Fz_val_rep,
        Fz_test_rep,
        Fz_overall_rep,
    )

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


def print_dataset_info(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    print("Dataset summary:")
    print(f"  Train: X shape={X_train.shape}, y shape={y_train.shape}")
    print(f"  Validation: X shape={X_val.shape}, y shape={y_val.shape}")
    print(f"  Test: X shape={X_test.shape}, y shape={y_test.shape}")


def build_regressor(
    num_variables,
    params,
    cfgfile,
    train_Fz_rep,
    scaler_X,
    scaler_y,
):
    regressor_params, config = load_config_data(cfgfile)
    regressor_params["num_individuals"] = params["num_individuals"]
    regressor_params["num_islands"] = params["num_islands"]

    batch_size = config["gp"]["batch_size"]
    penalty = config["gp"]["penalty"]
    fitness_scale = 1.0
    common_params = {
        "penalty": penalty,
        "fitness_scale": fitness_scale,
        "train_Fz_rep": train_Fz_rep,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }

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
        batch_size=batch_size,
        num_cpus=num_cpus,
        print_log=True,
        **regressor_params,
    )


# Custom optimization routine for Optuna.
# Although OptunaSearchCV could be used for this example,
# in practice one often needs a custom objective function.
# This shows how to define one and integrate it with Flex.
def optimize(
    trial,
    X,
    y,
    X_val,
    val_y,
    val_Fz_rep,
    train_Fz_rep,
    scaler_X,
    scaler_y,
    grid_search_parameters,
    cfgfile,
):
    num_variables = X.shape[1]
    params = {}
    for key in grid_search_parameters.keys():
        params[key] = trial.suggest_categorical(key, grid_search_parameters[key])

    gpsr = build_regressor(
        num_variables,
        params,
        cfgfile,
        train_Fz_rep,
        scaler_X,
        scaler_y,
    )
    gpsr.fit(X, y)
    validation_predictions = predict_force_model_with_regressor(
        gpsr,
        X_val,
        val_Fz_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )
    validation_score = r2_score(val_y, validation_predictions)
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
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler_X,
        scaler_y,
        num_variables,
        _,
        train_Fz_rep,
        val_Fz_rep,
        test_Fz_rep,
        Fz_overall_rep,
    ) = generate_dataset(scaleXy=scaleXy)

    print_dataset_info(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    if args.hpo:
        sampler = TPESampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            partial(
                optimize,
                X=X_train,
                y=y_train,
                X_val=X_val,
                val_y=y_val,
                val_Fz_rep=val_Fz_rep,
                train_Fz_rep=train_Fz_rep,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                grid_search_parameters=grid_search_parameters,
                cfgfile=str(CONFIG_PATH),
            ),
            n_trials=20,
        )
        best_params = study.best_trial.params
        best_validation_score = study.best_trial.value
    else:
        best_params = {
            "num_individuals": regressor_params["num_individuals"],
            "num_islands": regressor_params["num_islands"],
        }
        best_gpsr = build_regressor(
            num_variables,
            best_params,
            str(CONFIG_PATH),
            train_Fz_rep,
            scaler_X,
            scaler_y,
        )
        best_gpsr.fit(X_train, y_train)
        best_validation_predictions = predict_force_model_with_regressor(
            best_gpsr,
            X_val,
            val_Fz_rep,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
        )
        best_validation_score = r2_score(
            y_val,
            best_validation_predictions,
        )

    if args.hpo:
        best_gpsr = build_regressor(
            num_variables,
            best_params,
            str(CONFIG_PATH),
            train_Fz_rep,
            scaler_X,
            scaler_y,
        )
        best_gpsr.fit(X_train, y_train)

    save_model_results(
        best_gpsr,
        best_validation_score,
        best_params,
        X_train,
        y_train,
        train_Fz_rep,
        X_val,
        y_val,
        val_Fz_rep,
        X_test,
        y_test,
        test_Fz_rep,
        scaler_X,
        scaler_y,
        predict_force_model_with_regressor,
    )

    X_plot = np.concatenate([X_train[:, 0], X_val[:, 0], X_test[:, 0]])
    y_plot = np.concatenate([y_train, y_val, y_test])
    plot_force_model_data(
        X_plot,
        y_plot,
        Fz_rep=Fz_overall_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        V=MODEL_V,
        n_v=MODEL_N_V,
        n_x=MODEL_N_X,
        mu_expression=make_mu_expression_from_regressor(best_gpsr),
        output_path=PLOT_PATH,
        show=False,
    )

    print("Accuracy: {}".format(best_validation_score))
    print("Best hyperparameters: {}".format(best_params))
    print("Best model results saved to {}".format(RESULTS_PATH))
    print("Best model plot saved to {}".format(PLOT_PATH))


if __name__ == "__main__":
    main()
