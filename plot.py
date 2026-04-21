import matplotlib.pyplot as plt
import numpy as np

from force import compute_force_model
from process_data import load_and_process_dataset


DEFAULT_THETA_OPT = np.array([
    0.0668, 0.0001, 360.4850, 0.0230,
    0.6456, 3.07e-05,
    1.7213, 3.6888, 1.3652,
])


def plot_force_model_data(
    x_data,
    y_data,
    Fz_rep,
    theta_opt,
    V=16,
    n_v=200,
    n_x=100,
    mu_expression=None,
    colors=None,
    xlabel="Relative velocity v (m/s)",
    ylabel="Lateral force Fy (kN)",
):
    model_kwargs = {
        "theta_opt": theta_opt,
        "V": V,
        "n_v": n_v,
        "n_x": n_x,
    }

    if mu_expression is not None:
        model_kwargs["mu_expression"] = mu_expression

    if colors is None:
        colors = [
            (0.55, 0.80, 0.95),
            (1.00, 0.75, 0.65),
            (0.95, 0.85, 0.65),
            (0.75, 0.65, 0.75),
            (0.80, 0.90, 0.75),
        ]

    v, F_b = compute_force_model(Fz_rep, **model_kwargs)

    plt.figure()

    for k, (x_bin, y_bin) in enumerate(zip(x_data, y_data)):
        if len(x_bin) == 0:
            continue

        plt.plot(
            -x_bin * V,
            y_bin * Fz_rep[k] / 1000,
            "o",
            color=colors[k % len(colors)],
            markersize=4,
        )

    for k in range(len(Fz_rep)):
        plt.plot(v, F_b[k] / 1000, linewidth=1, label=f"Fz bin {k + 1}")

    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y, metadata = load_and_process_dataset(data_type=1, return_metadata=True)
    plot_force_model_data(
        X,
        y,
        metadata["Fz_rep"],
        theta_opt=DEFAULT_THETA_OPT,
    )
