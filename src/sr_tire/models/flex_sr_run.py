import argparse
import random
from copy import deepcopy
from functools import partial
from pathlib import Path

from flex.gp.util import (
    compile_individual_with_consts,
    load_config_data,
)
from flex.gp import regressor as gps
from flex.gp.primitives import add_primitives_to_pset_from_dict
import numpy as np
from sklearn.metrics import r2_score
from deap import gp
import optuna
from optuna.samplers import TPESampler
import ray

from ..force import MODEL_V
from ..plot import plot_force_model_data, plot_mu_curve
from ..process_data import load_and_process_bins, make_datasets
from .fitness import (
    assign_attributes,
    compute_force_model_MSE,
    compute_attributes,
    eval_model,
    predict,
    predict_force_model_with_regressor,
    score,
)
from .save_results import RESULTS_PATH, save_model_results

# set up number of cpus per ray worker
num_cpus = 1
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
PLOT_PATH = Path(__file__).resolve().with_name("best_model_plot.png")
MU_PLOT_PATH = Path(__file__).resolve().with_name("best_mu_plot.png")
RUN_RESULTS_DIR = Path(__file__).resolve().with_name("flex_runs")
RUN_SUMMARY_PATH = RUN_RESULTS_DIR / "summary.txt"
RUN_SUMMARY_LATEX_PATH = RUN_RESULTS_DIR / "summary.tex"


def generate_dataset():
    np.random.seed(42)
    num_variables = 1

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = make_datasets()
    _, _, Fz_overall_rep = load_and_process_bins()
    Fz_train_rep = Fz_overall_rep[:3]
    Fz_val_rep = Fz_overall_rep[3:4]
    Fz_test_rep = Fz_overall_rep[4:5]

    X_train = X_train.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    num_variables = X_train.shape[1]

    num_train_points = X_train.shape[0]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
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
        v_input = v.reshape(-1, 1)
        mu = eval_model(individual, v_input, consts)
        mu = mu.reshape(-1)
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
    custom_logger=None,
):
    regressor_params, config = load_config_data(cfgfile)
    regressor_params["num_individuals"] = params["num_individuals"]
    regressor_params["num_islands"] = params["num_islands"]

    batch_size = config["gp"]["batch_size"]
    penalty = config["gp"]["penalty"]
    fitness_scale = 1000.0
    common_params = {
        "penalty": penalty,
        "fitness_scale": fitness_scale,
        "train_Fz_rep": train_Fz_rep,
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
        custom_logger=custom_logger,
        remove_init_duplicates=True,
        **regressor_params,
    )


def optimize(
    trial,
    X,
    y,
    X_val,
    val_y,
    val_Fz_rep,
    train_Fz_rep,
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
    )
    gpsr.fit(X, y)
    validation_predictions = predict_force_model_with_regressor(
        gpsr,
        X_val,
        val_Fz_rep,
    )
    validation_score = r2_score(val_y, validation_predictions)
    return validation_score


def set_fit_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def fit_regressor(
    num_variables,
    params,
    cfgfile,
    train_Fz_rep,
    X_train,
    y_train,
    X_val,
    y_val,
    val_Fz_rep,
    seed,
):
    set_fit_seed(seed)
    train_mse_history = []
    val_mse_history = []

    gpsr = build_regressor(
        num_variables,
        params,
        cfgfile,
        train_Fz_rep,
    )
    toolbox, _ = gpsr._GPSymbolicRegressor__creator_toolbox_pset_config()

    def custom_logger(best_inds):
        best_individual = best_inds[0]
        train_mse_history.append(best_individual.train_mse)
        compiled_individual, _ = compile_individual_with_consts(best_individual, toolbox)
        consts = getattr(best_individual, "consts", [])
        val_mse = compute_force_model_MSE(
            compiled_individual,
            X_val[:, 0],
            y_val,
            val_Fz_rep,
            consts=consts,
        )
        val_mse_history.append(val_mse)

    gpsr.custom_logger = custom_logger
    gpsr.fit(X_train, y_train)
    validation_predictions = predict_force_model_with_regressor(
        gpsr,
        X_val,
        val_Fz_rep,
    )
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
        run_dir / "best_model_plot.png",
        run_dir / "best_mu_plot.png",
        run_dir / "train_mse_history.csv",
        run_dir / "val_mse_history.csv",
    )


def save_run_outputs(
    run_index,
    gpsr,
    validation_score,
    params,
    train_mse_history,
    val_mse_history,
    X_train,
    y_train,
    train_Fz_rep,
    X_val,
    y_val,
    val_Fz_rep,
    X_test,
    y_test,
    test_Fz_rep,
    X_plot,
    y_plot,
    Fz_overall_rep,
):
    (
        results_path,
        plot_path,
        mu_plot_path,
        train_mse_history_path,
        val_mse_history_path,
    ) = build_run_paths(run_index)
    result_data = save_model_results(
        gpsr,
        validation_score,
        params,
        X_train,
        y_train,
        train_Fz_rep,
        X_val,
        y_val,
        val_Fz_rep,
        X_test,
        y_test,
        test_Fz_rep,
        predict_force_model_with_regressor,
        results_path=results_path,
    )

    plot_force_model_data(
        X_plot,
        y_plot,
        Fz_rep=Fz_overall_rep,
        V=MODEL_V,
        mu_expression=make_mu_expression_from_regressor(gpsr),
        x_limits=(-5, 5),
        output_path=plot_path,
        show=False,
    )
    plot_mu_curve(
        mu_expression=make_mu_expression_from_regressor(gpsr),
        x_limits=(-5, 5),
        output_path=mu_plot_path,
        show=False,
    )
    train_mse_history_path.parent.mkdir(parents=True, exist_ok=True)
    history_lines = ["generation,train_mse"]
    history_lines.extend(
        f"{generation},{train_mse}"
        for generation, train_mse in enumerate(train_mse_history, start=1)
    )
    train_mse_history_path.write_text("\n".join(history_lines) + "\n")
    val_history_lines = ["generation,val_mse"]
    val_history_lines.extend(
        f"{generation},{val_mse}"
        for generation, val_mse in enumerate(val_mse_history, start=1)
    )
    val_mse_history_path.write_text("\n".join(val_history_lines) + "\n")

    return {
        "run_index": run_index,
        "validation_score": validation_score,
        "results_path": results_path,
        "plot_path": plot_path,
        "mu_plot_path": mu_plot_path,
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
                f"  mu_plot_path: {run_summary['mu_plot_path']}",
                f"  train_mse_history_path: {run_summary['train_mse_history_path']}",
                f"  val_mse_history_path: {run_summary['val_mse_history_path']}",
                "",
            ]
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n")


def save_run_summary_latex(
    run_summaries,
    latex_path=RUN_SUMMARY_LATEX_PATH,
):
    if not run_summaries:
        raise ValueError("run_summaries must contain at least one run")

    metric_keys = [
        ("train_r2", r"R^2 train"),
        ("test_r2", r"R^2 test"),
        ("train_rmse", "RMSE train"),
        ("test_rmse", "RMSE test"),
    ]
    median_metrics = {
        key: float(np.median([run_summary[key] for run_summary in run_summaries]))
        for key, _ in metric_keys
    }
    best_test_run = max(run_summaries, key=lambda run_summary: run_summary["test_r2"])

    def format_metric(value):
        return f"{value:.6f}"

    table_rows = [
        "Median"
        + "".join(f" & {format_metric(median_metrics[key])}" for key, _ in metric_keys)
        + r" \\",
        f"Best test $R^2$ (run {best_test_run['run_index']:03d})"
        + "".join(
            f" & {format_metric(best_test_run[key])}" for key, _ in metric_keys
        )
        + r" \\",
    ]
    header_cells = "Statistic" + "".join(
        f" & ${label}$" if "R^2" in label else f" & {label}"
        for _, label in metric_keys
    )
    latex_lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        r"\section*{Flex SR Run Summary}",
        f"Total runs: {len(run_summaries)}\\\\",
        f"Best test $R^2$ run: {best_test_run['run_index']:03d}\\\\",
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        header_cells + r" \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Median metrics across repeated Flex SR runs and the metrics of the run with the best test $R^2$.}",
        r"\end{table}",
        r"\end{document}",
    ]
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(latex_lines) + "\n")


def main():
    if not ray.is_initialized():
        ray.init(runtime_env={"working_dir": str(ROOT_DIR)})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Enable Optuna hyperparameter optimization.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Override the number of repeated flex runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for repeated runs.",
    )
    args = parser.parse_args()

    grid_search_parameters = {
        "num_individuals": [250],
        "num_islands": [1],
    }

    regressor_params, _ = load_config_data(str(CONFIG_PATH))
    num_runs = args.num_runs
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    base_seed = args.seed

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        num_variables,
        _,
        train_Fz_rep,
        val_Fz_rep,
        test_Fz_rep,
        Fz_overall_rep,
    ) = generate_dataset()

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
    X_plot = np.concatenate([X_train[:, 0], X_val[:, 0], X_test[:, 0]])
    y_plot = np.concatenate([y_train, y_val, y_test])
    run_summaries = []
    best_run = None
    best_gpsr = None

    for run_index in range(1, num_runs + 1):
        run_seed = base_seed + run_index - 1
        print(f"Starting run {run_index}/{num_runs} with seed {run_seed}")
        gpsr, validation_score, train_mse_history, val_mse_history = fit_regressor(
            num_variables,
            best_params,
            str(CONFIG_PATH),
            train_Fz_rep,
            X_train,
            y_train,
            X_val,
            y_val,
            val_Fz_rep,
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
            train_Fz_rep,
            X_val,
            y_val,
            val_Fz_rep,
            X_test,
            y_test,
            test_Fz_rep,
            X_plot,
            y_plot,
            Fz_overall_rep,
        )
        run_summary["seed"] = run_seed
        run_summaries.append(run_summary)

        if best_run is None or validation_score > best_run["validation_score"]:
            best_run = run_summary
            best_gpsr = gpsr

    save_run_summary(run_summaries)
    save_run_summary_latex(run_summaries)

    save_model_results(
        best_gpsr,
        best_run["validation_score"],
        deepcopy(best_params),
        X_train,
        y_train,
        train_Fz_rep,
        X_val,
        y_val,
        val_Fz_rep,
        X_test,
        y_test,
        test_Fz_rep,
        predict_force_model_with_regressor,
    )

    plot_force_model_data(
        X_plot,
        y_plot,
        Fz_rep=Fz_overall_rep,
        V=MODEL_V,
        mu_expression=make_mu_expression_from_regressor(best_gpsr),
        x_limits=(-5, 5),
        output_path=PLOT_PATH,
        show=False,
    )
    plot_mu_curve(
        mu_expression=make_mu_expression_from_regressor(best_gpsr),
        x_limits=(-5, 5),
        output_path=MU_PLOT_PATH,
        show=False,
    )

    print("Accuracy: {}".format(best_run["validation_score"]))
    print("Best hyperparameters: {}".format(best_params))
    print("Best run: {}".format(best_run["run_index"]))
    print("Best model results saved to {}".format(RESULTS_PATH))
    print("Best model plot saved to {}".format(PLOT_PATH))
    print("Best mu(v) plot saved to {}".format(MU_PLOT_PATH))
    print("Per-run results saved under {}".format(RUN_RESULTS_DIR))
    print("Run summary saved to {}".format(RUN_SUMMARY_PATH))
    print("LaTeX run summary saved to {}".format(RUN_SUMMARY_LATEX_PATH))


if __name__ == "__main__":
    main()
