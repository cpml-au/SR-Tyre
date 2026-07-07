import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from deap import base, creator, gp, tools
from flex.gp.numpy_primitives import conversion_rules
from flex.gp.primitives import add_primitives_to_pset_from_dict
from flex.gp.sympy import stringify_for_sympy
from flex.gp.util import compile_individual_with_consts, load_config_data
import numpy as np
from sklearn.metrics import r2_score
import sympy as sp

from ..force import MODEL_V
from ..plot import plot_force_model_data, plot_mu_curve
from .fitness import (
    compute_force_model_MSE,
    eval_MSE_and_tune_constants,
    eval_model,
    predict_force_model_from_callable,
    normalize_force_predictions,
)
from .flex_sr_run import generate_dataset, print_dataset_info
from .save_results import STEADY_STRIBECK_OUTPUT_DIR, save_model_results


CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
RESULTS_PATH = STEADY_STRIBECK_OUTPUT_DIR / "random_search_best_model_results.txt"
PLOT_PATH = STEADY_STRIBECK_OUTPUT_DIR / "random_search_best_model_plot.png"
MU_PLOT_PATH = STEADY_STRIBECK_OUTPUT_DIR / "random_search_best_mu_plot.png"
RUN_RESULTS_DIR = STEADY_STRIBECK_OUTPUT_DIR / "random_search_runs"
RUN_SUMMARY_PATH = RUN_RESULTS_DIR / "summary.txt"
RUN_SUMMARY_LATEX_PATH = RUN_RESULTS_DIR / "summary.tex"

DEFAULT_TIME_BUDGET_SECONDS = 420.0


def format_sympy_expression(expression):
    return str(sp.sympify(expression)).replace("ARG0", "v")


@dataclass
class RandomSearchModel:
    best_individual: gp.PrimitiveTree
    toolbox: base.Toolbox

    def get_best_individual_sympy(self):
        expression = stringify_for_sympy(
            self.best_individual,
            conversion_rules,
            "c",
        )
        return format_sympy_expression(expression)


def set_search_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_toolbox(num_variables, cfgfile):
    _, config = load_config_data(cfgfile)
    pset = gp.PrimitiveSetTyped("Main", [float] * num_variables, float)
    add_primitives_to_pset_from_dict(pset, config["gp"]["primitives"])
    if config["gp"]["use_constants"]:
        pset.addTerminal(object, float, "c")

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=config["gp"]["min_"],
        max_=config["gp"]["max_"],
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", lambda individual: creator.Individual(individual))
    return toolbox, config


def predict_force_model_with_random_search(model, X, Fz_rep):
    individual, _ = compile_individual_with_consts(model.best_individual, model.toolbox)
    consts = getattr(model.best_individual, "consts", [])
    force_pred = predict_force_model_from_callable(
        individual,
        X,
        Fz_rep,
        consts=consts,
    )
    return normalize_force_predictions(force_pred, Fz_rep)


def make_mu_expression_from_random_search(model):
    individual, _ = compile_individual_with_consts(model.best_individual, model.toolbox)
    consts = getattr(model.best_individual, "consts", [])

    def mu_expression(v, mu_s, v_s, delta_s):
        v_input = v.reshape(-1, 1)
        mu = eval_model(individual, v_input, consts=consts)
        if np.isscalar(mu) or np.asarray(mu).size == 1:
            mu = np.full(v_input.shape[0], float(np.asarray(mu).reshape(-1)[0]))
        return np.nan_to_num(
            np.asarray(mu).reshape(-1),
            nan=1.0,
            posinf=1e8,
            neginf=-1e8,
        )

    return mu_expression


def evaluate_candidate(tree, toolbox, X_train, y_train, train_Fz_rep):
    if len(tree) >= 50:
        return np.inf, []

    try:
        mse, consts = eval_MSE_and_tune_constants(
            tree,
            toolbox,
            X_train[:, 0],
            y_train,
            train_Fz_rep,
        )
    except Exception:
        mse = np.inf
        consts = []

    if not np.isfinite(mse):
        mse = np.inf
    return float(mse), consts


def compute_validation_mse(model, X_val, y_val, val_Fz_rep):
    individual, _ = compile_individual_with_consts(model.best_individual, model.toolbox)
    consts = getattr(model.best_individual, "consts", [])
    return compute_force_model_MSE(
        individual,
        X_val[:, 0],
        y_val,
        val_Fz_rep,
        consts=consts,
    )


def run_random_search(
    num_variables,
    cfgfile,
    X_train,
    y_train,
    train_Fz_rep,
    X_val,
    y_val,
    val_Fz_rep,
    seed,
    time_budget_seconds,
    max_samples=None,
):
    set_search_seed(seed)
    toolbox, config = build_toolbox(num_variables, cfgfile)

    deadline = time.monotonic() + time_budget_seconds
    start_time = time.monotonic()
    best_tree = None
    best_train_mse = np.inf
    best_val_mse = np.inf
    train_mse_history = []
    val_mse_history = []
    samples_evaluated = 0

    while time.monotonic() < deadline:
        if max_samples is not None and samples_evaluated >= max_samples:
            break

        tree = toolbox.individual()
        train_mse, consts = evaluate_candidate(
            tree,
            toolbox,
            X_train,
            y_train,
            train_Fz_rep,
        )
        samples_evaluated += 1

        if train_mse < best_train_mse:
            tree.consts = consts
            tree.train_mse = train_mse
            best_tree = tree
            best_train_mse = train_mse
            model = RandomSearchModel(best_tree, toolbox)
            best_val_mse = compute_validation_mse(
                model,
                X_val,
                y_val,
                val_Fz_rep,
            )

        train_mse_history.append(best_train_mse)
        val_mse_history.append(best_val_mse)

        if samples_evaluated % 10 == 0:
            elapsed = time.monotonic() - start_time
            print(
                "sample {} elapsed {:.1f}s best_train_mse {:.6g}".format(
                    samples_evaluated,
                    elapsed,
                    best_train_mse,
                )
            )

    if best_tree is None:
        raise RuntimeError("Random search did not produce a finite candidate")

    elapsed_seconds = time.monotonic() - start_time
    model = RandomSearchModel(best_tree, toolbox)
    validation_predictions = predict_force_model_with_random_search(
        model,
        X_val,
        val_Fz_rep,
    )
    validation_score = r2_score(y_val, validation_predictions)
    search_params = {
        "time_budget_seconds": time_budget_seconds,
        "elapsed_seconds": elapsed_seconds,
        "evaluated_samples": samples_evaluated,
        "max_samples": max_samples,
        "min_": config["gp"]["min_"],
        "max_": config["gp"]["max_"],
        "primitives": [entry["name"] for entry in config["gp"]["primitives"]["used"]],
    }
    return (
        model,
        validation_score,
        np.asarray(train_mse_history, dtype=float),
        np.asarray(val_mse_history, dtype=float),
        search_params,
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


def write_history(path, column_name, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"sample,{column_name}"]
    lines.extend(f"{sample},{value}" for sample, value in enumerate(values, start=1))
    path.write_text("\n".join(lines) + "\n")


def save_run_outputs(
    run_index,
    model,
    validation_score,
    search_params,
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
        model,
        validation_score,
        search_params,
        X_train,
        y_train,
        train_Fz_rep,
        X_val,
        y_val,
        val_Fz_rep,
        X_test,
        y_test,
        test_Fz_rep,
        predict_force_model_with_random_search,
        results_path=results_path,
    )

    mu_expression = make_mu_expression_from_random_search(model)
    plot_force_model_data(
        X_plot,
        y_plot,
        Fz_rep=Fz_overall_rep,
        V=MODEL_V,
        mu_expression=mu_expression,
        x_limits=(-5, 5),
        output_path=plot_path,
        show=False,
    )
    plot_mu_curve(
        mu_expression=mu_expression,
        x_limits=(-5, 5),
        output_path=mu_plot_path,
        show=False,
    )
    write_history(train_mse_history_path, "best_train_mse", train_mse_history)
    write_history(val_mse_history_path, "best_val_mse", val_mse_history)

    return {
        "run_index": run_index,
        "validation_score": validation_score,
        "results_path": results_path,
        "plot_path": plot_path,
        "mu_plot_path": mu_plot_path,
        "train_mse_history_path": train_mse_history_path,
        "val_mse_history_path": val_mse_history_path,
        "elapsed_seconds": search_params["elapsed_seconds"],
        "evaluated_samples": search_params["evaluated_samples"],
        **result_data,
    }


def save_run_summary(run_summaries, summary_path=RUN_SUMMARY_PATH):
    summary_lines = []
    for run_summary in run_summaries:
        summary_lines.extend(
            [
                f"run_{run_summary['run_index']:03d}",
                f"  seed: {run_summary['seed']}",
                f"  elapsed_seconds: {run_summary['elapsed_seconds']}",
                f"  evaluated_samples: {run_summary['evaluated_samples']}",
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


def save_run_summary_latex(run_summaries, latex_path=RUN_SUMMARY_LATEX_PATH):
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
        r"\section*{Random Search Run Summary}",
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
        r"\caption{Median metrics across repeated random-search runs and the metrics of the run with the best test $R^2$.}",
        r"\end{table}",
        r"\end{document}",
    ]
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(latex_lines) + "\n")


def save_best_overall_outputs(
    model,
    validation_score,
    search_params,
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
    save_model_results(
        model,
        validation_score,
        search_params,
        X_train,
        y_train,
        train_Fz_rep,
        X_val,
        y_val,
        val_Fz_rep,
        X_test,
        y_test,
        test_Fz_rep,
        predict_force_model_with_random_search,
        results_path=RESULTS_PATH,
    )
    mu_expression = make_mu_expression_from_random_search(model)
    plot_force_model_data(
        X_plot,
        y_plot,
        Fz_rep=Fz_overall_rep,
        V=MODEL_V,
        mu_expression=mu_expression,
        x_limits=(-5, 5),
        output_path=PLOT_PATH,
        show=False,
    )
    plot_mu_curve(
        mu_expression=mu_expression,
        x_limits=(-5, 5),
        output_path=MU_PLOT_PATH,
        show=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of independent random-search runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for repeated runs.",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
        help="Wall-clock random-search budget per run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on evaluated random expressions, mainly for smoke tests.",
    )
    args = parser.parse_args()

    if args.num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    if args.time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be at least 1 when provided")

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

    X_plot = np.concatenate([X_train[:, 0], X_val[:, 0], X_test[:, 0]])
    y_plot = np.concatenate([y_train, y_val, y_test])
    run_summaries = []
    best_run = None
    best_model = None
    best_search_params = None

    for run_index in range(1, args.num_runs + 1):
        run_seed = args.seed + run_index - 1
        print(
            "Starting random-search run {}/{} with seed {} and {:.1f}s budget".format(
                run_index,
                args.num_runs,
                run_seed,
                args.time_budget_seconds,
            )
        )
        (
            model,
            validation_score,
            train_mse_history,
            val_mse_history,
            search_params,
        ) = run_random_search(
            num_variables,
            str(CONFIG_PATH),
            X_train,
            y_train,
            train_Fz_rep,
            X_val,
            y_val,
            val_Fz_rep,
            run_seed,
            args.time_budget_seconds,
            max_samples=args.max_samples,
        )
        run_summary = save_run_outputs(
            run_index,
            model,
            validation_score,
            search_params,
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
            best_model = model
            best_search_params = search_params

    save_run_summary(run_summaries)
    save_run_summary_latex(run_summaries)
    save_best_overall_outputs(
        best_model,
        best_run["validation_score"],
        best_search_params,
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

    print("Accuracy: {}".format(best_run["validation_score"]))
    print("Best run: {}".format(best_run["run_index"]))
    print("Evaluated samples in best run: {}".format(best_run["evaluated_samples"]))
    print("Elapsed seconds in best run: {}".format(best_run["elapsed_seconds"]))
    print("Best random-search model results saved to {}".format(RESULTS_PATH))
    print("Best random-search model plot saved to {}".format(PLOT_PATH))
    print("Best random-search mu(v) plot saved to {}".format(MU_PLOT_PATH))
    print("Per-run random-search results saved under {}".format(RUN_RESULTS_DIR))
    print("Run summary saved to {}".format(RUN_SUMMARY_PATH))
    print("LaTeX run summary saved to {}".format(RUN_SUMMARY_LATEX_PATH))


if __name__ == "__main__":
    main()
