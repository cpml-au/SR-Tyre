import warnings

from flex.gp.util import (
    compile_individual_with_consts,
    detect_nested_trigonometric_functions,
)
import numpy as np
import pygmo as pg
from sklearn.metrics import r2_score

from ..force import MODEL_V, compute_force_model
from ..process_data import reshape_flattened_bins


PSO_GENERATIONS = 20
PSO_SWARM_SIZE = 50


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
    consts=[],
):
    def mu_expression(v, mu_s, v_s, delta_s):
        v_input = v.reshape(-1, 1)
        mu = eval_model(individual, v_input, consts)
        return mu.reshape(v.shape)

    X_bins = reshape_flattened_bins(X, Fz_rep)
    v_bins = -X_bins * MODEL_V
    _, F_b = compute_force_model(
        Fz_rep,
        V=MODEL_V,
        mu_expression=mu_expression,
        v=v_bins,
    )
    return F_b.reshape(-1)


def normalize_force_predictions(force_pred, Fz_rep):
    force_bins = reshape_flattened_bins(force_pred, Fz_rep)
    return (force_bins / Fz_rep[:, None]).reshape(-1)


def predict_force_model_with_regressor(gpsr, X, Fz_rep):
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()
    individual, _ = compile_individual_with_consts(gpsr._best, toolbox)
    consts = getattr(gpsr._best, "consts", [])
    force_pred = predict_force_model_from_callable(
        individual,
        X,
        Fz_rep,
        consts=consts,
    )
    return normalize_force_predictions(force_pred, Fz_rep)


def compute_force_model_MSE(
    individual,
    X,
    y,
    Fz_rep,
    consts=[],
):
    force_pred = predict_force_model_from_callable(
        individual,
        X,
        Fz_rep,
        consts=consts,
    )
    y_pred = normalize_force_predictions(force_pred, Fz_rep)
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
):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)

    if num_consts > 0:
        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err = compute_force_model_MSE(
                    individual,
                    X,
                    y,
                    Fz_rep,
                    consts=x,
                )
                return [total_err]

            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.pso(gen=PSO_GENERATIONS))
        pop = pg.population(prb, size=PSO_SWARM_SIZE)
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
