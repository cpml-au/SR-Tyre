import warnings
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygmo as pg
from dctkit.dec import cochain as C
from flex.gp.util import (
    compile_individual_with_consts,
    detect_nested_trigonometric_functions,
)
from sklearn.metrics import r2_score

from sr_tire.dynamic_pde.compare_pacejka import (
    DEFAULT_AMPLITUDE_RATIO,
    DEFAULT_DATASET_PATH,
    DEFAULT_EXCITATION,
    DEFAULT_OMEGA,
    DEFAULT_PARAMETERIZATION,
    DEFAULT_RUN,
    DEFAULT_SIGMA_0,
    load_pacejka_fx_dataset,
)
from sr_tire.dynamic_pde.simulate_pde import TirePDEParameters
from sr_tire.dynamic_pde.simulate_pde_dctkit import (
    DctkitLineOperators,
    simulate_pde_dctkit_with_rhs_term,
)


INVALID_MSE = 1.0e6
DEFAULT_PSO_GENERATIONS = 8
DEFAULT_PSO_SWARM_SIZE = 12


def cap_fitness_value(value, cap=INVALID_MSE):
    if value is None or np.isnan(value) or np.isinf(value):
        return cap
    return min(float(value), cap)


class JittedCochainIndividual:
    """JIT a Flex individual over cochain coefficient arrays."""

    def __init__(self, individual):
        self.individual = individual
        self._compiled_by_complex = {}

    def _compiled_for_complex(self, complex_):
        key = id(complex_)
        if key not in self._compiled_by_complex:

            def evaluate_coeffs(z_coeffs, sigma_coeffs, xi_coeffs, consts):
                z_p0 = C.CochainP0(complex_, z_coeffs)
                sigma_p0 = C.CochainP0(complex_, sigma_coeffs)
                xi_p0 = C.CochainP0(complex_, xi_coeffs)
                value = self.individual(z_p0, sigma_p0, xi_p0, consts)
                if isinstance(value, C.Cochain):
                    return value.coeffs.reshape(-1)
                return jnp.asarray(value)

            self._compiled_by_complex[key] = jax.jit(evaluate_coeffs)
        return self._compiled_by_complex[key]

    def __call__(self, z_p0, sigma_p0, xi_p0, consts=None):
        if consts is None:
            consts = []
        evaluate_coeffs = self._compiled_for_complex(z_p0.complex)
        return evaluate_coeffs(
            z_p0.coeffs.reshape(-1),
            sigma_p0.coeffs.reshape(-1),
            xi_p0.coeffs.reshape(-1),
            jnp.asarray(consts, dtype=float),
        )


def jit_individual(individual):
    if isinstance(individual, JittedCochainIndividual):
        return individual
    return JittedCochainIndividual(individual)


@dataclass(frozen=True)
class DynamicPDEDataset:
    """Trajectory and PDE settings used to score learned RHS terms."""

    t: np.ndarray
    force: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    parameterization: int
    excitation: str
    run: int
    params: TirePDEParameters
    n_x: int
    sigma_0: float
    omega: float
    amplitude_ratio: float
    force_scale: float
    clear_jax_caches: bool
    solver_max_dt: float | None


def _split_indices(num_points, train_fraction=0.7, val_fraction=0.15):
    indices = np.arange(num_points, dtype=int)
    train_end = max(1, int(round(train_fraction * num_points)))
    val_end = max(train_end + 1, int(round((train_fraction + val_fraction) * num_points)))
    val_end = min(val_end, num_points - 1)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _build_index_features(indices):
    return np.asarray(indices, dtype=float).reshape(-1, 1)


def load_dynamic_pde_dataset(
    dataset_path=DEFAULT_DATASET_PATH,
    parameterization=DEFAULT_PARAMETERIZATION,
    excitation=DEFAULT_EXCITATION,
    run=DEFAULT_RUN,
    params=TirePDEParameters(),
    n_x=50,
    sigma_0=DEFAULT_SIGMA_0,
    omega=DEFAULT_OMEGA,
    amplitude_ratio=DEFAULT_AMPLITUDE_RATIO,
    train_fraction=0.7,
    val_fraction=0.15,
    force_scale="auto",
    time_stride=1,
    max_time_points=None,
    clear_jax_caches=True,
    solver_max_dt=None,
):
    """Load one CSV force trajectory and split its time indices."""

    parameterizations, excitations, runs, t, _sigma, force = load_pacejka_fx_dataset(
        Path(dataset_path)
    )
    mask = (
        (parameterizations == int(parameterization))
        & (excitations == excitation)
        & (runs == int(run))
    )
    if not np.any(mask):
        raise ValueError(
            "No force trajectory found for "
            f"parameterization={parameterization}, excitation={excitation}, run={run}."
        )

    t = np.asarray(t, dtype=float)
    force_row = np.asarray(force[mask][0], dtype=float)
    time_stride = int(time_stride)
    if time_stride < 1:
        raise ValueError("time_stride must be at least 1.")
    if max_time_points is not None:
        max_time_points = int(max_time_points)
        if max_time_points < 2:
            raise ValueError("max_time_points must be at least 2.")
        time_stride = max(time_stride, int(np.ceil(force_row.size / max_time_points)))

    if time_stride > 1:
        t = t[::time_stride]
        force_row = force_row[::time_stride]

    train_indices, val_indices, test_indices = _split_indices(
        force_row.size,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    if force_scale == "auto":
        scale = float(np.std(force_row[train_indices]))
        if not np.isfinite(scale) or scale < 1.0:
            scale = max(float(np.sqrt(np.mean(force_row[train_indices] ** 2))), 1.0)
    else:
        scale = float(force_scale)

    dataset = DynamicPDEDataset(
        t=t,
        force=force_row,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        parameterization=int(parameterization),
        excitation=str(excitation),
        run=int(run),
        params=params,
        n_x=int(n_x),
        sigma_0=float(sigma_0),
        omega=float(omega),
        amplitude_ratio=float(amplitude_ratio),
        force_scale=scale,
        clear_jax_caches=bool(clear_jax_caches),
        solver_max_dt=None if solver_max_dt is None else float(solver_max_dt),
    )

    return (
        _build_index_features(train_indices),
        force_row[train_indices],
        _build_index_features(val_indices),
        force_row[val_indices],
        _build_index_features(test_indices),
        force_row[test_indices],
        dataset,
    )


def make_rhs_term_from_callable(individual, consts=None):
    individual = jit_individual(individual)
    if consts is None:
        consts = []

    def rhs_term(
        t,
        z_p0,
        sigma_p0,
        xi_p0,
        params: TirePDEParameters,
        operators: DctkitLineOperators,
    ):
        del t, params, operators
        warnings.filterwarnings("ignore")
        return individual(z_p0, sigma_p0, xi_p0, consts)

    return rhs_term


def solve_force_with_rhs_callable(individual, dataset: DynamicPDEDataset, consts=None):
    rhs_term = make_rhs_term_from_callable(individual, consts=consts)
    n_t = dataset.t.size
    if dataset.solver_max_dt is not None:
        t_span_length = float(dataset.t[-1] - dataset.t[0])
        n_t = max(n_t, int(np.ceil(t_span_length / dataset.solver_max_dt)) + 1)

    result = simulate_pde_dctkit_with_rhs_term(
        params=dataset.params,
        n_x=dataset.n_x,
        t_span=(float(dataset.t[0]), float(dataset.t[-1])),
        n_t=n_t,
        sigma_0=dataset.sigma_0,
        omega=dataset.omega,
        amplitude_ratio=dataset.amplitude_ratio,
        rhs_term=rhs_term,
    )
    t_solve = np.asarray(jax.device_get(result.t), dtype=float)
    force_solve = np.asarray(jax.device_get(result.force), dtype=float)
    if t_solve.size == dataset.t.size and np.allclose(t_solve, dataset.t):
        return force_solve
    return np.interp(dataset.t, t_solve, force_solve)


def predict_force_from_callable(individual, X, dataset: DynamicPDEDataset, consts=None):
    indices = np.asarray(X[:, 0], dtype=int)
    force = solve_force_with_rhs_callable(individual, dataset, consts=consts)
    return force[indices]


def compute_dynamic_pde_force_MSE(
    individual,
    X,
    y,
    dataset: DynamicPDEDataset,
    consts=None,
    clear_caches=None,
):
    if clear_caches is None:
        clear_caches = dataset.clear_jax_caches

    mse = INVALID_MSE
    try:
        y_pred = predict_force_from_callable(individual, X, dataset, consts=consts)
        residual = (np.asarray(y, dtype=float) - y_pred) / dataset.force_scale
        mse = float(np.mean(residual**2))
    except Exception:
        mse = INVALID_MSE
    finally:
        if clear_caches:
            jax.clear_caches()

    return cap_fitness_value(mse)


def eval_MSE_and_tune_constants(
    tree,
    toolbox,
    X,
    y,
    dataset: DynamicPDEDataset,
    constant_bounds=(-1.0, 1.0),
    pso_generations=DEFAULT_PSO_GENERATIONS,
    pso_swarm_size=DEFAULT_PSO_SWARM_SIZE,
):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)
    individual = jit_individual(individual)

    if num_consts > 0:
        x0 = np.zeros(num_consts)
        lower, upper = constant_bounds
        if int(pso_generations) <= 0 or int(pso_swarm_size) <= 0:
            mse = compute_dynamic_pde_force_MSE(
                individual,
                X,
                y,
                dataset,
                consts=x0,
            )
            return mse, x0

        class fitting_problem:
            def fitness(self, x):
                total_err = compute_dynamic_pde_force_MSE(
                    individual,
                    X,
                    y,
                    dataset,
                    consts=x,
                    clear_caches=False,
                )
                return [total_err]

            def get_bounds(self):
                return (lower * np.ones(num_consts), upper * np.ones(num_consts))

        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.pso(gen=int(pso_generations)))
        pop = pg.population(prb, size=int(pso_swarm_size))
        pop.set_x(0, x0)
        pop = algo.evolve(pop)
        if dataset.clear_jax_caches:
            jax.clear_caches()
        mse = float(pop.champion_f[0])
        consts = pop.champion_x

        mse = cap_fitness_value(mse)
    else:
        mse = compute_dynamic_pde_force_MSE(individual, X, y, dataset)
        consts = []
    return mse, consts


def get_features_batch(individuals_batch, individ_feature_extractors=None):
    if individ_feature_extractors is None:
        individ_feature_extractors = [
            len,
            lambda ind: detect_nested_trigonometric_functions(str(ind)),
        ]

    features_batch = [
        [extractor(ind) for ind in individuals_batch]
        for extractor in individ_feature_extractors
    ]
    return features_batch[0], features_batch[1]


def predict(individuals_batch, toolbox, X, penalty, fitness_scale, dataset):
    del penalty, fitness_scale
    predictions = [None] * len(individuals_batch)
    for i, tree in enumerate(individuals_batch):
        callable_individual, _ = compile_individual_with_consts(tree, toolbox)
        callable_individual = jit_individual(callable_individual)
        predictions[i] = predict_force_from_callable(
            callable_individual,
            X,
            dataset,
            consts=getattr(tree, "consts", []),
        )
    return predictions


def compute_attributes(
    individuals_batch,
    toolbox,
    X,
    y,
    penalty,
    fitness_scale,
    dataset,
    max_tree_length=40,
    constant_bounds=(-1.0, 1.0),
    pso_generations=DEFAULT_PSO_GENERATIONS,
    pso_swarm_size=DEFAULT_PSO_SWARM_SIZE,
):
    attributes = [None] * len(individuals_batch)
    individ_length, nested_trigs = get_features_batch(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        if individ_length[i] >= max_tree_length:
            consts = None
            mse = INVALID_MSE
            fitness = (INVALID_MSE,)
        else:
            mse, consts = eval_MSE_and_tune_constants(
                tree,
                toolbox,
                X,
                y,
                dataset,
                constant_bounds=constant_bounds,
                pso_generations=pso_generations,
                pso_swarm_size=pso_swarm_size,
            )
            fitness = (
                cap_fitness_value(
                    fitness_scale
                    * (
                        mse
                        + 100000.0 * nested_trigs[i]
                        + penalty["reg_param"] * individ_length[i]
                    )
                ),
            )
        attributes[i] = {"consts": consts, "fitness": fitness, "train_mse": mse}
    return attributes


def assign_attributes(individuals_batch, attributes):
    for ind, attr in zip(individuals_batch, attributes):
        ind.consts = attr["consts"]
        ind.train_mse = attr["train_mse"]
        ind.fitness.values = attr["fitness"]


def score(individuals_batch, toolbox, X, y, penalty, fitness_scale, dataset):
    predictions = predict(individuals_batch, toolbox, X, penalty, fitness_scale, dataset)
    scores = [None] * len(individuals_batch)
    for i, prediction in enumerate(predictions):
        try:
            scores[i] = r2_score(y, prediction)
        except Exception:
            scores[i] = -np.inf
    return scores
