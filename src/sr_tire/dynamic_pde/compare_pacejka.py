import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from sr_tire.dynamic_pde.simulate_pde import (
    PROJECT_ROOT,
    TirePDEParameters,
    simulate_pde,
)


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "Fx_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "pde"
DEFAULT_PARAMETERIZATION = 1
DEFAULT_EXCITATION = "var2"
DEFAULT_RUN = 5
DEFAULT_SIGMA_0 = 0.12
DEFAULT_OMEGA = 10.0 * jnp.pi
DEFAULT_AMPLITUDE_RATIO = 0.4


class PacejkaComparison(NamedTuple):
    parameterization: np.ndarray
    excitation: np.ndarray
    run: np.ndarray
    t: np.ndarray
    sigma: np.ndarray
    pde_sigma: np.ndarray
    pacejka_force: np.ndarray
    pde_force: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    max_abs_error: np.ndarray
    sigma_0: float
    omega: float
    amplitude_ratio: float


def load_pacejka_fx_dataset(path=DEFAULT_DATASET_PATH):
    """Load and group Pacejka force trajectories from ``Fx_dataset.csv``."""

    groups = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                int(row["parameterization"]),
                row["excitation"],
                int(row["run"]),
            )
            groups.setdefault(key, []).append(
                (float(row["t"]), float(row["sigma"]), float(row["Fx"]))
            )

    keys = sorted(groups)
    t_ref = None
    sigma_rows = []
    force_rows = []
    parameterization = []
    excitation = []
    run = []

    for key in keys:
        rows = np.array(sorted(groups[key]), dtype=float)
        t = rows[:, 0]
        if t_ref is None:
            t_ref = t
        elif not np.allclose(t_ref, t):
            raise ValueError("All Pacejka trajectories must share the same t grid.")

        parameterization.append(key[0])
        excitation.append(key[1])
        run.append(key[2])
        sigma_rows.append(rows[:, 1])
        force_rows.append(rows[:, 2])

    return (
        np.asarray(parameterization),
        np.asarray(excitation),
        np.asarray(run),
        t_ref,
        np.vstack(sigma_rows),
        np.vstack(force_rows),
    )


def compare_with_pacejka_dataset(
    dataset_path=DEFAULT_DATASET_PATH,
    params=TirePDEParameters(),
    n_x=100,
    parameterization_id=DEFAULT_PARAMETERIZATION,
    excitation_name=DEFAULT_EXCITATION,
    run_id=DEFAULT_RUN,
    sigma_0=DEFAULT_SIGMA_0,
    omega=DEFAULT_OMEGA,
    amplitude_ratio=DEFAULT_AMPLITUDE_RATIO,
):
    """Compare PDE forces to the matching Pacejka trajectory.

    The intended comparison is parameterization 1, excitation ``var2``, run 5.
    In ``Fx_dataset.csv`` that trajectory is the sinusoid
    ``0.12 + 0.4*0.12*sin(10*pi*t)`` over ``t in [0, 1]``.
    """

    parameterization, excitation, run, t, sigma, pacejka_force = load_pacejka_fx_dataset(
        dataset_path
    )
    mask = (
        (parameterization == parameterization_id)
        & (excitation == excitation_name)
        & (run == run_id)
    )
    if not np.any(mask):
        raise ValueError(
            "No Pacejka trajectory found for "
            f"parameterization={parameterization_id}, excitation={excitation_name}, "
            f"run={run_id}."
        )

    parameterization = parameterization[mask]
    excitation = excitation[mask]
    run = run[mask]
    sigma = sigma[mask]
    pacejka_force = pacejka_force[mask]

    result = simulate_pde(
        params=params,
        n_x=n_x,
        t_span=(float(t[0]), float(t[-1])),
        n_t=t.size,
        sigma_0=sigma_0,
        omega=omega,
        amplitude_ratio=amplitude_ratio,
    )
    pde_force = np.asarray(jax.device_get(result.force))[np.newaxis, :]
    error = pde_force - pacejka_force

    return PacejkaComparison(
        parameterization=parameterization,
        excitation=excitation,
        run=run,
        t=t,
        sigma=sigma,
        pde_sigma=np.asarray(jax.device_get(result.slip))[np.newaxis, :],
        pacejka_force=pacejka_force,
        pde_force=pde_force,
        rmse=np.sqrt(np.mean(error**2, axis=1)),
        mae=np.mean(np.abs(error), axis=1),
        max_abs_error=np.max(np.abs(error), axis=1),
        sigma_0=float(sigma_0),
        omega=float(omega),
        amplitude_ratio=float(amplitude_ratio),
    )


def write_pacejka_comparison_summary(comparison, output_path):
    """Write per-trajectory error metrics to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "parameterization",
                "excitation",
                "run",
                "sigma_0",
                "omega",
                "amplitude_ratio",
                "rmse",
                "mae",
                "max_abs_error",
            ]
        )
        for row in zip(
            comparison.parameterization,
            comparison.excitation,
            comparison.run,
            np.full_like(comparison.rmse, comparison.sigma_0),
            np.full_like(comparison.rmse, comparison.omega),
            np.full_like(comparison.rmse, comparison.amplitude_ratio),
            comparison.rmse,
            comparison.mae,
            comparison.max_abs_error,
        ):
            writer.writerow(row)


def print_pacejka_comparison_summary(comparison):
    """Print compact aggregate comparison metrics."""

    total_error = comparison.pde_force - comparison.pacejka_force
    print("PDE vs. Pacejka Fx comparison")
    print(f"Trajectories: {comparison.rmse.size}")
    print(
        "PDE slip input: "
        f"sigma_0={comparison.sigma_0:.6g}, "
        f"amplitude_ratio={comparison.amplitude_ratio:.6g}, "
        f"omega={comparison.omega:.6g}"
    )
    print(f"Overall RMSE: {float(np.sqrt(np.mean(total_error**2))):.6g} N")
    print(f"Median trajectory RMSE: {float(np.median(comparison.rmse)):.6g} N")
    print(f"Worst trajectory RMSE: {float(np.max(comparison.rmse)):.6g} N")

    for excitation in np.unique(comparison.excitation):
        mask = comparison.excitation == excitation
        error = comparison.pde_force[mask] - comparison.pacejka_force[mask]
        print(f"{excitation}: RMSE={float(np.sqrt(np.mean(error**2))):.6g} N")


def find_trajectory_index(comparison, excitation=DEFAULT_EXCITATION, run=DEFAULT_RUN):
    """Return the index of one selected Pacejka/PDE trajectory pair."""

    mask = (comparison.excitation == excitation) & (comparison.run == run)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"No trajectory found for excitation={excitation}, run={run}.")
    return int(indices[0])


def plot_pacejka_comparison(
    comparison,
    output_dir=None,
    show=True,
    excitation=DEFAULT_EXCITATION,
    run=DEFAULT_RUN,
):
    """Plot one selected PDE/Pacejka force pair and their slip inputs."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    idx = find_trajectory_index(comparison, excitation, run)
    parameterization = int(comparison.parameterization[idx])
    fig, (ax_force, ax_slip) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 5.4),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_force.plot(
        comparison.t,
        comparison.pacejka_force[idx],
        linewidth=1.2,
        label="Pacejka",
    )
    ax_force.plot(
        comparison.t,
        comparison.pde_force[idx],
        linestyle="--",
        linewidth=1.2,
        label="PDE",
    )
    ax_force.set_title(
        f"parameterization={parameterization}, excitation={excitation}, run={run}, "
        f"sigma_0={comparison.sigma_0:g}, "
        f"amp={comparison.amplitude_ratio:g}, omega={comparison.omega:g}"
    )
    ax_force.set_ylabel(r"Tire force $F_x$ (N)")
    ax_force.grid(True)
    ax_force.legend()

    ax_slip.plot(
        comparison.t,
        comparison.sigma[idx],
        linewidth=1.2,
        label="CSV slip",
    )
    ax_slip.plot(
        comparison.t,
        comparison.pde_sigma[idx],
        linestyle="--",
        linewidth=1.2,
        label="Analytical PDE slip",
    )
    ax_slip.set_xlabel(r"Travelled distance $s$ (m)")
    ax_slip.set_ylabel(r"Slip $\sigma$ (-)")
    ax_slip.grid(True)
    ax_slip.legend()

    if output_dir is not None:
        fig.savefig(output_dir / "pacejka_comparison.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PDE tire forces against Pacejka Fx trajectories."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Pacejka comparison dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for comparison plots and metrics. Use an empty string to skip saving.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot windows after writing results.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compute metrics without creating comparison plots.",
    )
    parser.add_argument(
        "--excitation",
        default=DEFAULT_EXCITATION,
        help="Excitation name to plot, for example const, var1, or var2.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=DEFAULT_RUN,
        help="Run number to plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    comparison = compare_with_pacejka_dataset(
        args.dataset,
        excitation_name=args.excitation,
        run_id=args.run,
    )
    print_pacejka_comparison_summary(comparison)

    if output_dir is not None:
        write_pacejka_comparison_summary(
            comparison,
            output_dir / "pacejka_comparison_metrics.csv",
        )

    if not args.no_plots:
        plot_pacejka_comparison(
            comparison,
            output_dir=output_dir,
            show=args.show,
            excitation=args.excitation,
            run=args.run,
        )
        if output_dir is not None:
            print(f"Saved comparison plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
