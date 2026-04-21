from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score


RESULTS_PATH = Path(__file__).resolve().with_name("best_model_results.txt")


def compute_regression_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


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
    scaler_X,
    scaler_y,
    predict_force_model_with_regressor,
    results_path=RESULTS_PATH,
):
    best_model = str(gpsr.get_best_individual_sympy())

    y_train_pred = predict_force_model_with_regressor(
        gpsr,
        X_train,
        train_Fz_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )
    y_val_pred = predict_force_model_with_regressor(
        gpsr,
        X_val,
        val_Fz_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )
    y_test_pred = predict_force_model_with_regressor(
        gpsr,
        X_test,
        test_Fz_rep,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )

    y_train_true = np.asarray(train_y)
    y_val_true = np.asarray(val_y)
    y_test_true = np.asarray(test_y)

    train_rmse, train_r2 = compute_regression_metrics(y_train_true, y_train_pred)
    val_rmse, val_r2 = compute_regression_metrics(y_val_true, y_val_pred)
    test_rmse, test_r2 = compute_regression_metrics(y_test_true, y_test_pred)

    result_lines = [
        f"best_model: {best_model}",
        f"best_validation_score: {validation_score}",
        f"best_hyperparameters: {params}",
        f"train_rmse: {train_rmse}",
        f"train_r2: {train_r2}",
        f"val_rmse: {val_rmse}",
        f"val_r2: {val_r2}",
        f"test_rmse: {test_rmse}",
        f"test_r2: {test_r2}",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(result_lines) + "\n")
