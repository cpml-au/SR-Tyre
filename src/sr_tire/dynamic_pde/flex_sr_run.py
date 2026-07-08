import argparse
import random
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import ray
from deap import gp
from dctkit.dec import cochain as C
from flex.gp import regressor as gps
from flex.gp.primitives import add_primitives_to_pset_from_dict
from flex.gp.util import compile_individual_with_consts, load_config_data
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score

from sr_tire.dynamic_pde.fitness import (
    assign_attributes,
    compute_attributes,
    compute_dynamic_pde_force_MSE,
    load_dynamic_pde_dataset,
    predict,
    predict_force_from_callable,
    score,
    solve_force_with_rhs_callable,
)
from sr_tire.dynamic_pde.simulate_pde import PROJECT_ROOT, TirePDEParameters


num_cpus = 1
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
DYNAMIC_PDE_OUTPUT_DIR = PROJECT_ROOT / "results" / "dynamic-pde"
RESULTS_PATH = DYNAMIC_PDE_OUTPUT_DIR / "best_model_results.txt"
PLOT_PATH = DYNAMIC_PDE_OUTPUT_DIR / "best_force_plot.png"
RUN_RESULTS_DIR = DYNAMIC_PDE_OUTPUT_DIR / "flex_runs"
RUN_SUMMARY_PATH = RUN_RESULTS_DIR / "summary.txt"


def set_fit_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _read_dataset_config(config):
    cfg = config.get("dynamic_pde", {})
    params_cfg = cfg.get("params", {})
    params_cfg = {key: float(value) for key, value in params_cfg.items()}
    params = TirePDEParameters(**params_cfg)
    dataset_path = Path(cfg.get("dataset_path", PROJECT_ROOT / "data" / "Fx_dataset.csv"))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return {
        "dataset_path": dataset_path,
        "parameterization": cfg.get("parameterization", 1),
        "excitation": cfg.get("excitation", "var2"),
        "run": cfg.get("run", 5),
        "params": params,
        "n_x": cfg.get("n_x", 50),
        "sigma_0": cfg.get("sigma_0", 0.12),
        "omega": cfg.get("omega", 10.0 * np.pi),
        "amplitude_ratio": cfg.get("amplitude_ratio", 0.4),
        "train_fraction": cfg.get("train_fraction", 0.7),
        "val_fraction": cfg.get("val_fraction", 0.15),
        "force_scale": cfg.get("force_scale", "auto"),
    }


def generate_dataset(config):
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dataset,
    ) = load_dynamic_pde_dataset(**_read_dataset_config(config))
    return X_train, y_train, X_val, y_val, X_test, y_test, dataset


def print_dataset_info(X_train, y_train, X_val, y_val, X_test, y_test, dataset):
    print("Dynamic PDE dataset summary:")
    print(
        "  "
        f"parameterization={dataset.parameterization}, "
        f"excitation={dataset.excitation}, run={dataset.run}"
    )
    print(f"  t grid: {dataset.t.size} points from {dataset.t[0]} to {dataset.t[-1]}")
    print(
        "  PDE excitation: "
        f"sigma_0={dataset.sigma_0}, "
        f"amplitude_ratio={dataset.amplitude_ratio}, omega={dataset.omega}"
    )
    print(f"  Train: X shape={X_train.shape}, y shape={y_train.shape}")
    print(f"  Validation: X shape={X_val.shape}, y shape={y_val.shape}")
    print(f"  Test: X shape={X_test.shape}, y shape={y_test.shape}")


def build_regressor(num_variables, params, cfgfile, dataset, custom_logger=None):
    if num_variables != 3:
        raise ValueError("Dynamic PDE RHS search expects z, sigma, and xi inputs.")

    regressor_params, config = load_config_data(cfgfile)
    regressor_params["num_individuals"] = params["num_individuals"]
    regressor_params["num_islands"] = params["num_islands"]

    gp_config = config["gp"]
    fit_config = config.get("fitness", {})
    common_params = {
        "penalty": gp_config["penalty"],
        "fitness_scale": fit_config.get("fitness_scale", 1.0),
        "dataset": dataset,
        "max_tree_length": fit_config.get("max_tree_length", 40),
        "constant_bounds": tuple(fit_config.get("constant_bounds", [-1.0, 1.0])),
        "pso_generations": fit_config.get("pso_generations", 8),
        "pso_swarm_size": fit_config.get("pso_swarm_size", 12),
    }

    pset = gp.PrimitiveSetTyped(
        "Main",
        [C.CochainP0, C.CochainP0, C.CochainP0],
        C.CochainP0,
    )
    pset.renameArguments(ARG0="z", ARG1="sigma", ARG2="xi")
    pset = add_primitives_to_pset_from_dict(pset, gp_config["primitives"])
    if gp_config["use_constants"]:
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
        batch_size=gp_config["batch_size"],
        num_cpus=gp_config.get("num_cpus", num_cpus),
        print_log=True,
        custom_logger=custom_logger,
        remove_init_duplicates=True,
        **regressor_params,
    )


def predict_dynamic_force_with_regressor(gpsr, X, dataset):
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()
    individual, _ = compile_individual_with_consts(gpsr._best, toolbox)
    consts = getattr(gpsr._best, "consts", [])
    return predict_force_from_callable(individual, X, dataset, consts=consts)


def compute_regression_metrics(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


def get_best_model_expression(gpsr):
    try:
        best_model = str(gpsr.get_best_individual_sympy())
    except Exception:
        best_model = str(gpsr._best)
        consts = getattr(gpsr._best, "consts", [])
        if len(consts) > 0:
            best_model = f"{best_model} ; consts={np.asarray(consts).tolist()}"
    return (
        best_model.replace("ARG0", "z")
        .replace("ARG1", "sigma")
        .replace("ARG2", "xi")
    )


def collect_model_results(
    gpsr,
    validation_score,
    params,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dataset,
):
    y_train_pred = predict_dynamic_force_with_regressor(gpsr, X_train, dataset)
    y_val_pred = predict_dynamic_force_with_regressor(gpsr, X_val, dataset)
    y_test_pred = predict_dynamic_force_with_regressor(gpsr, X_test, dataset)

    train_rmse, train_r2 = compute_regression_metrics(y_train, y_train_pred)
    val_rmse, val_r2 = compute_regression_metrics(y_val, y_val_pred)
    test_rmse, test_r2 = compute_regression_metrics(y_test, y_test_pred)

    return {
        "best_model": get_best_model_expression(gpsr),
        "best_validation_score": validation_score,
        "best_hyperparameters": params,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "force_scale": dataset.force_scale,
    }


def save_model_results(
    gpsr,
    validation_score,
    params,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dataset,
    results_path=RESULTS_PATH,
):
    result_data = collect_model_results(
        gpsr,
        validation_score,
        params,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dataset,
    )
    result_lines = [
        f"best_model: {result_data['best_model']}",
        f"best_validation_score: {result_data['best_validation_score']}",
        f"best_hyperparameters: {result_data['best_hyperparameters']}",
        f"train_rmse: {result_data['train_rmse']}",
        f"train_r2: {result_data['train_r2']}",
        f"val_rmse: {result_data['val_rmse']}",
        f"val_r2: {result_data['val_r2']}",
        f"test_rmse: {result_data['test_rmse']}",
        f"test_r2: {result_data['test_r2']}",
        f"force_scale: {result_data['force_scale']}",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(result_lines) + "\n")
    return result_data


def plot_force_trajectory(gpsr, dataset, output_path=PLOT_PATH, show=False):
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()
    individual, _ = compile_individual_with_consts(gpsr._best, toolbox)
    consts = getattr(gpsr._best, "consts", [])
    force_pred = solve_force_with_rhs_callable(individual, dataset, consts=consts)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(dataset.t, dataset.force, linewidth=1.2, label="CSV force")
    ax.plot(dataset.t, force_pred, linewidth=1.1, linestyle="--", label="PDE + SR RHS")
    ax.set_xlabel(r"Travelled distance $s$ (m)")
    ax.set_ylabel(r"Tire force $F_x$ (N)")
    ax.grid(True)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def optimize(
    trial,
    X,
    y,
    X_val,
    val_y,
    dataset,
    grid_search_parameters,
    cfgfile,
):
    params = {
        key: trial.suggest_categorical(key, values)
        for key, values in grid_search_parameters.items()
    }
    gpsr = build_regressor(3, params, cfgfile, dataset)
    gpsr.fit(X, y)
    validation_predictions = predict_dynamic_force_with_regressor(gpsr, X_val, dataset)
    return r2_score(val_y, validation_predictions)


def fit_regressor(
    params,
    cfgfile,
    X_train,
    y_train,
    X_val,
    y_val,
    dataset,
    seed,
):
    set_fit_seed(seed)
    train_mse_history = []
    val_mse_history = []

    gpsr = build_regressor(3, params, cfgfile, dataset)
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()

    def custom_logger(best_inds):
        best_individual = best_inds[0]
        train_mse_history.append(best_individual.train_mse)
        compiled_individual, _ = compile_individual_with_consts(best_individual, toolbox)
        val_mse = compute_dynamic_pde_force_MSE(
            compiled_individual,
            X_val,
            y_val,
            dataset,
            consts=getattr(best_individual, "consts", []),
        )
        val_mse_history.append(val_mse)

    gpsr.custom_logger = custom_logger
    gpsr.fit(X_train, y_train)
    validation_predictions = predict_dynamic_force_with_regressor(gpsr, X_val, dataset)
    validation_score = r2_score(y_val, validation_predictions)
    return (
        gpsr,
        validation_score,
        np.asarray(train_mse_history, dtype=float),
        np.asarray(val_mse_history, dtype=float),
    )


def build_run_paths(run_index):
    run_dir = RUN_RESULTS_DIR / f"run_{run_index:03d}"
    return (
        run_dir / "best_model_results.txt",
        run_dir / "best_force_plot.png",
        run_dir / "train_mse_history.csv",
        run_dir / "val_mse_history.csv",
    )


def _write_history(path, name, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"generation,{name}"]
    lines.extend(f"{generation},{value}" for generation, value in enumerate(values, 1))
    path.write_text("\n".join(lines) + "\n")


def save_run_outputs(
    run_index,
    gpsr,
    validation_score,
    params,
    train_mse_history,
    val_mse_history,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dataset,
):
    results_path, plot_path, train_mse_history_path, val_mse_history_path = (
        build_run_paths(run_index)
    )
    result_data = save_model_results(
        gpsr,
        validation_score,
        params,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dataset,
        results_path=results_path,
    )
    plot_force_trajectory(gpsr, dataset, output_path=plot_path, show=False)
    _write_history(train_mse_history_path, "train_mse", train_mse_history)
    _write_history(val_mse_history_path, "val_mse", val_mse_history)

    return {
        "run_index": run_index,
        "validation_score": validation_score,
        "results_path": results_path,
        "plot_path": plot_path,
        "train_mse_history_path": train_mse_history_path,
        "val_mse_history_path": val_mse_history_path,
        **result_data,
    }


def save_run_summary(run_summaries, summary_path=RUN_SUMMARY_PATH):
    summary_lines = []
    for run_summary in run_summaries:
        summary_lines.extend(
            [
                f"run_{run_summary['run_index']:03d}",
                f"  seed: {run_summary['seed']}",
                f"  best_validation_score: {run_summary['validation_score']}",
                f"  train_r2: {run_summary['train_r2']}",
                f"  train_rmse: {run_summary['train_rmse']}",
                f"  test_r2: {run_summary['test_r2']}",
                f"  test_rmse: {run_summary['test_rmse']}",
                f"  best_model: {run_summary['best_model']}",
                f"  results_path: {run_summary['results_path']}",
                f"  plot_path: {run_summary['plot_path']}",
                f"  train_mse_history_path: {run_summary['train_mse_history_path']}",
                f"  val_mse_history_path: {run_summary['val_mse_history_path']}",
                "",
            ]
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Flex SR for an additive dynamic-PDE RHS term."
    )
    parser.add_argument("--hpo", action="store_true", help="Enable Optuna HPO.")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    if not ray.is_initialized():
        ray.init(runtime_env={"working_dir": str(ROOT_DIR)})

    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("num-runs must be at least 1")

    regressor_params, config = load_config_data(str(CONFIG_PATH))
    X_train, y_train, X_val, y_val, X_test, y_test, dataset = generate_dataset(config)
    print_dataset_info(X_train, y_train, X_val, y_val, X_test, y_test, dataset)

    grid_search_parameters = {
        "num_individuals": [regressor_params["num_individuals"]],
        "num_islands": [regressor_params["num_islands"]],
    }
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
                dataset=dataset,
                grid_search_parameters=grid_search_parameters,
                cfgfile=str(CONFIG_PATH),
            ),
            n_trials=20,
        )
        best_params = study.best_trial.params
    else:
        best_params = {
            "num_individuals": regressor_params["num_individuals"],
            "num_islands": regressor_params["num_islands"],
        }

    run_summaries = []
    best_run = None
    best_gpsr = None

    for run_index in range(1, args.num_runs + 1):
        run_seed = args.seed + run_index - 1
        print(f"Starting run {run_index}/{args.num_runs} with seed {run_seed}")
        gpsr, validation_score, train_mse_history, val_mse_history = fit_regressor(
            deepcopy(best_params),
            str(CONFIG_PATH),
            X_train,
            y_train,
            X_val,
            y_val,
            dataset,
            run_seed,
        )
        run_summary = save_run_outputs(
            run_index,
            gpsr,
            validation_score,
            deepcopy(best_params),
            train_mse_history,
            val_mse_history,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            dataset,
        )
        run_summary["seed"] = run_seed
        run_summaries.append(run_summary)
        if best_run is None or validation_score > best_run["validation_score"]:
            best_run = run_summary
            best_gpsr = gpsr

    save_run_summary(run_summaries)
    save_model_results(
        best_gpsr,
        best_run["validation_score"],
        deepcopy(best_params),
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dataset,
    )
    plot_force_trajectory(best_gpsr, dataset, output_path=PLOT_PATH, show=False)

    print("Accuracy: {}".format(best_run["validation_score"]))
    print("Best hyperparameters: {}".format(best_params))
    print("Best run: {}".format(best_run["run_index"]))
    print("Best model results saved to {}".format(RESULTS_PATH))
    print("Best force plot saved to {}".format(PLOT_PATH))
    print("Per-run results saved under {}".format(RUN_RESULTS_DIR))
    print("Run summary saved to {}".format(RUN_SUMMARY_PATH))


if __name__ == "__main__":
    main()
