import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sr_tire.force import DEFAULT_THETA_OPT, compute_force_model
from sr_tire.process_data import (
    load_and_process_bins,
    make_datasets,
    reshape_flattened_bins,
)


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
        _, _, Fz_rep = load_and_process_bins()

    Fz_rep = np.atleast_1d(np.asarray(Fz_rep, dtype=float)).reshape(-1)
    X_bins = reshape_flattened_bins(X_plot, Fz_rep)
    y_bins = reshape_flattened_bins(y_plot, Fz_rep)

    model_kwargs = {
        "theta_opt": theta_opt,
        "V": V,
        "n_v": n_v,
        "n_x": n_x,
    }

    if mu_expression is not None:
        model_kwargs["mu_expression"] = mu_expression

    v, F_b = compute_force_model(Fz_rep, **model_kwargs)

    plt.figure()
    colors = [color] if Fz_rep.size == 1 else plt.cm.viridis(np.linspace(0.15, 0.85, Fz_rep.size))
    for i, Fz_bin in enumerate(Fz_rep):
        plt.plot(
            -X_bins[i] * V,
            y_bins[i] * Fz_bin / 1000,
            "o",
            color=colors[i],
            markersize=4,
            label=f"Data (Fz={Fz_bin:.1f} N)",
        )
        plt.plot(
            v,
            F_b[i] / 1000,
            linewidth=1,
            color=colors[i],
            label=f"Model (Fz={Fz_bin:.1f} N)",
        )

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
    X_train, y_train, X_val, y_val, X_test, y_test = make_datasets(data_type=1)
    _, _, Fz_all = load_and_process_bins(data_type=1)
    X = np.concatenate([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_val, y_test])

    plot_force_model_data(
        X,
        y,
        Fz_rep=Fz_all,
        theta_opt=DEFAULT_THETA_OPT,
    )


if __name__ == "__main__":
    main()
