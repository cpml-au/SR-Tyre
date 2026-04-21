import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .force import DEFAULT_THETA_OPT, compute_force_model
from .process_data import load_and_process_dataset, load_split_representative_loads


def plot_force_model_data(
    X,
    y,
    Fz_rep=None,
    theta_opt=None,
    scaler_X=None,
    scaler_y=None,
    V=16,
    n_v=200,
    n_x=100,
    mu_expression=None,
    color=(0.55, 0.80, 0.95),
    xlabel="Relative velocity v (m/s)",
    ylabel="Lateral force Fy (kN)",
    output_path=None,
    show=True,
):
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if scaler_X is not None:
        X_plot = scaler_X.inverse_transform(X.reshape(-1, 1)).reshape(-1)
    else:
        X_plot = X

    if scaler_y is not None:
        y_plot = scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    else:
        y_plot = y

    if Fz_rep is None:
        _, _, _, Fz_rep = load_split_representative_loads()

    model_kwargs = {
        "theta_opt": theta_opt,
        "V": V,
        "n_v": n_v,
        "n_x": n_x,
    }

    if mu_expression is not None:
        model_kwargs["mu_expression"] = mu_expression

    v, F_b = compute_force_model(np.array([Fz_rep]), **model_kwargs)

    plt.figure()
    plt.plot(
        -X_plot * V,
        y_plot * Fz_rep / 1000,
        "o",
        color=color,
        markersize=4,
        label="Data",
    )
    plt.plot(v, F_b[0] / 1000, linewidth=1, label=f"Model (Fz={Fz_rep:.1f} N)")

    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_dataset(data_type=1)
    _, _, _, Fz_overall_rep = load_split_representative_loads(data_type=1)
    X = np.concatenate([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_val, y_test])

    plot_force_model_data(
        X,
        y,
        Fz_rep=Fz_overall_rep,
        theta_opt=DEFAULT_THETA_OPT,
    )


if __name__ == "__main__":
    main()
