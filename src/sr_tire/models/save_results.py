from pathlib import Path

from flex.gp.numpy_primitives import conversion_rules
from flex.gp.sympy import stringify_for_sympy
import numpy as np
from sklearn.metrics import r2_score


RESULTS_PATH = Path(__file__).resolve().with_name("best_model_results.txt")


def compute_regression_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def get_best_model_expression(gpsr):
    try:
        best_model = gpsr.get_best_individual_sympy()
    except Exception:
        try:
            best_model = stringify_for_sympy(gpsr._best, conversion_rules, "c")
        except Exception:
            best_model = str(gpsr._best)
            consts = getattr(gpsr._best, "consts", [])
            if len(consts) > 0:
                best_model = f"{best_model} ; consts={np.asarray(consts).tolist()}"
    return str(best_model).replace("ARG0", "v")


def collect_model_results(
    gpsr,
    validation_score,
    params,
    X_train,
    train_y,
    train_Fz_rep,
    X_val,
    val_y,
    val_Fz_rep,
    X_test,
    test_y,
    test_Fz_rep,
    predict_force_model_with_regressor,
):
    best_model = get_best_model_expression(gpsr)

    y_train_pred = predict_force_model_with_regressor(
        gpsr,
        X_train,
        train_Fz_rep,
    )
    y_val_pred = predict_force_model_with_regressor(
        gpsr,
        X_val,
        val_Fz_rep,
    )
    y_test_pred = predict_force_model_with_regressor(
        gpsr,
        X_test,
        test_Fz_rep,
    )

    y_train_true = train_y
    y_val_true = val_y
    y_test_true = test_y

    train_rmse, train_r2 = compute_regression_metrics(y_train_true, y_train_pred)
    val_rmse, val_r2 = compute_regression_metrics(y_val_true, y_val_pred)
    test_rmse, test_r2 = compute_regression_metrics(y_test_true, y_test_pred)

    return {
        "best_model": best_model,
        "best_validation_score": validation_score,
        "best_hyperparameters": params,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    }


def format_model_results(result_data):
    return [
        f"best_model: {result_data['best_model']}",
        f"best_validation_score: {result_data['best_validation_score']}",
        f"best_hyperparameters: {result_data['best_hyperparameters']}",
        f"train_rmse: {result_data['train_rmse']}",
        f"train_r2: {result_data['train_r2']}",
        f"val_rmse: {result_data['val_rmse']}",
        f"val_r2: {result_data['val_r2']}",
        f"test_rmse: {result_data['test_rmse']}",
        f"test_r2: {result_data['test_r2']}",
    ]


def save_model_results(
    gpsr,
    validation_score,
    params,
    X_train,
    train_y,
    train_Fz_rep,
    X_val,
    val_y,
    val_Fz_rep,
    X_test,
    test_y,
    test_Fz_rep,
    predict_force_model_with_regressor,
    results_path=RESULTS_PATH,
):
    result_data = collect_model_results(
        gpsr,
        validation_score,
        params,
        X_train,
        train_y,
        train_Fz_rep,
        X_val,
        val_y,
        val_Fz_rep,
        X_test,
        test_y,
        test_Fz_rep,
        predict_force_model_with_regressor,
    )
    result_lines = format_model_results(result_data)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(result_lines) + "\n")
    return result_data
