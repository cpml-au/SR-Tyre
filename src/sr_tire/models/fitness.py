import warnings

from flex.gp.util import (
    compile_individual_with_consts,
    detect_nested_trigonometric_functions,
)
import numpy as np
import pygmo as pg
from sklearn.metrics import r2_score

from ..force import compute_force_model


MODEL_V = 16
MODEL_N_V = 200
MODEL_N_X = 100


def inverse_transform_features(X, scaler_X):
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    if scaler_X is None:
        return X[:, 0]
    return scaler_X.inverse_transform(X)[:, 0]


def eval_model(individual, X, consts=[]):
    num_variables = X.shape[1]
    X = [X[:, i] for i in range(num_variables)]
    warnings.filterwarnings("ignore")
    y_pred = individual(*X, consts)
    return y_pred


def predict_force_model_from_callable(
    individual,
    X,
    Fz_rep,
    scaler_X=None,
    scaler_y=None,
    consts=[],
):
    max_abs_prediction = 1e8

    def mu_expression(v, mu_s, v_s, delta_s):
        v_input = np.asarray(v).reshape(-1, 1)
        mu = eval_model(individual, v_input, consts)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        return np.nan_to_num(mu, nan=1.0, posinf=1e8, neginf=-1e8)

    v_model, F_b = compute_force_model(
        np.array([Fz_rep]),
        V=MODEL_V,
        n_v=MODEL_N_V,
        n_x=MODEL_N_X,
        mu_expression=mu_expression,
    )

    X_raw = inverse_transform_features(X, scaler_X)
    v_query = -X_raw * MODEL_V
    y_model = np.asarray(F_b[0] / Fz_rep, dtype=float)
    y_model = np.nan_to_num(
        y_model,
        nan=0.0,
        posinf=max_abs_prediction,
        neginf=-max_abs_prediction,
    )
    y_model = np.clip(y_model, -max_abs_prediction, max_abs_prediction)
    y_pred = np.interp(v_query, v_model, y_model)
    y_pred = np.nan_to_num(
        y_pred,
        nan=0.0,
        posinf=max_abs_prediction,
        neginf=-max_abs_prediction,
    )
    y_pred = np.clip(y_pred, -max_abs_prediction, max_abs_prediction)

    if scaler_y is None:
        return y_pred
    return scaler_y.transform(y_pred.reshape(-1, 1)).reshape(-1)


def predict_force_model_with_regressor(gpsr, X, Fz_rep, scaler_X=None, scaler_y=None):
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()
    individual, _ = compile_individual_with_consts(gpsr._best, toolbox)
    consts = getattr(gpsr._best, "consts", [])
    return predict_force_model_from_callable(
        individual,
        X,
        Fz_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        consts=consts,
    )


def compute_force_model_MSE(
    individual,
    X,
    y,
    Fz_rep,
    scaler_X=None,
    scaler_y=None,
    consts=[],
):
    y_pred = predict_force_model_from_callable(
        individual,
        X,
        Fz_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        consts=consts,
    )
    mse = np.mean((y - y_pred) ** 2)

    if np.isnan(mse) or np.isinf(mse):
        mse = 1e8

    return mse


def eval_MSE_and_tune_constants(
    tree,
    toolbox,
    X,
    y,
    Fz_rep,
    scaler_X=None,
    scaler_y=None,
):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)

    if num_consts > 0:
        x0 = np.ones(num_consts)
        swarm_size = 10

        class fitting_problem:
            def fitness(self, x):
                total_err = compute_force_model_MSE(
                    individual,
                    X,
                    y,
                    Fz_rep,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    consts=x,
                )
                return [total_err]

            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.pso(gen=10))
        pop = pg.population(prb, size=swarm_size)
        pop.set_x(0, x0)
        pop = algo.evolve(pop)
        mse = pop.champion_f[0]
        consts = pop.champion_x

        if np.isinf(mse) or np.isnan(mse):
            mse = 1e8
    else:
        mse = compute_force_model_MSE(
            individual,
            X,
            y,
            Fz_rep,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
        )
        consts = []
    return mse, consts


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


def compute_attributes(
    individuals_batch,
    toolbox,
    X,
    y,
    penalty,
    fitness_scale,
    train_Fz_rep,
    scaler_X,
    scaler_y,
):
    attributes = [None] * len(individuals_batch)

    individ_length, nested_trigs = get_features_batch(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        if individ_length[i] >= 50:
            consts = None
            fitness = (1e8,)
        else:
            mse, consts = eval_MSE_and_tune_constants(
                tree,
                toolbox,
                X[:, 0],
                y,
                train_Fz_rep,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
            )
            fitness = (
                fitness_scale
                * (
                    mse
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
