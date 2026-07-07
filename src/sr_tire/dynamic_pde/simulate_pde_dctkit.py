import argparse
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from dctkit.dec import cochain as C
from dctkit.mesh import util

from sr_tire.dynamic_pde.simulate_pde import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    SimulationResult,
    TirePDEParameters,
    plot_simulation,
    simulate_pde,
    slip_input,
    stribeck_mu,
    structural_coefficients,
)


ArrayLike: TypeAlias = jax.Array | np.ndarray
ScalarLike: TypeAlias = float | jax.Array
RhsArgs: TypeAlias = tuple[
    "TirePDEParameters",
    "DctkitLineOperators",
    ScalarLike,
    ScalarLike,
    ScalarLike,
]


class DctkitLineOperators(NamedTuple):
    """Cached DCTKit objects for the one-dimensional contact line."""

    complex: object
    dx: float
    ones_p0: C.Cochain


@lru_cache(maxsize=None)
def build_line_operators(n_x: int) -> DctkitLineOperators:
    """Build dctkit cochain operators on the unit contact interval.

    The bristle deflection ``z`` is stored as a primal 0-cochain because its
    values live at the contact-line grid points. Its spatial difference is a
    primal 1-cochain obtained with the coboundary operator. Applying the Hodge
    star to primal 0-cochains gives the dual-cell weights, so ``C.inner`` with a
    constant 0-cochain reproduces the trapezoidal integral used by the baseline
    simulator.
    """

    mesh, _ = util.generate_line_mesh(n_x, L=1.0, x_min=0.0)
    complex_ = util.build_complex_from_mesh(mesh, space_dim=1)
    complex_.get_hodge_star()
    xi = np.asarray(complex_.node_coords[:, 0])
    dx = float(xi[1] - xi[0])
    ones_p0 = C.CochainP0(complex_, np.ones(n_x))
    return DctkitLineOperators(complex=complex_, dx=dx, ones_p0=ones_p0)


def constant_p0(operators: DctkitLineOperators, value: ScalarLike) -> C.Cochain:
    """Return a primal 0-cochain with the same scalar value at every node."""

    return C.scalar_mul(operators.ones_p0, value)


def upwind_derivative_p0(
    z_p0: C.Cochain,
    operators: DctkitLineOperators,
) -> C.Cochain:
    """Return the upwind spatial derivative as a primal 0-cochain.

    ``C.coboundary(z_p0)`` is naturally a primal 1-cochain on edges. The tire
    model needs a nodal derivative in the same primal 0-space as ``z`` and uses
    the leading-edge boundary condition ``z(xi=0)=0``. This interpolation from
    edge differences back to upwind nodal values is the only coefficient-level
    step left in the DCTKit RHS.

    The Burgers tutorial can use ``flat_dual_upw`` directly because its unknown
    is a dual 0-cochain at cell/circumcenter locations and the update is written
    as a flux balance. Here ``z`` is kept as a primal 0-cochain at the contact
    nodes to match ``simulate_pde.py`` exactly, so the same flat-based upwind
    operator would require changing the state placement and the reference
    discretization.
    """

    dz_p1 = C.coboundary(z_p0)
    leading_edge_difference = z_p0.coeffs[:1]
    edge_differences = dz_p1.coeffs
    dzdx = jnp.concatenate((leading_edge_difference, edge_differences), axis=0)
    return C.scalar_mul(C.CochainP0(operators.complex, dzdx), 1.0 / operators.dx)


def compute_force_dctkit(
    z: jax.Array,
    params: TirePDEParameters,
    operators: DctkitLineOperators,
) -> jax.Array:
    """Integrate bristle deflection with DCTKit's cochain inner product."""

    def integrate_row(z_row: jax.Array) -> jax.Array:
        z_p0 = C.CochainP0(operators.complex, z_row)
        return C.inner(z_p0, operators.ones_p0)

    return params.Fz * params.k_0 * jax.vmap(integrate_row)(z)


def tire_pde_rhs_dctkit(
    t: ScalarLike,
    z: jax.Array,
    params: TirePDEParameters,
    operators: DctkitLineOperators,
    sigma_0: ScalarLike = 0.12,
    omega: ScalarLike = 5.0 * jnp.pi,
    amplitude_ratio: ScalarLike = 0.6,
) -> jax.Array:
    """Dynamic tire PDE RHS assembled with dctkit cochains."""

    z_p0 = C.CochainP0(operators.complex, z)
    dzdx_p0 = upwind_derivative_p0(z_p0, operators)
    integral_z = C.inner(z_p0, operators.ones_p0)
    zL = z_p0.coeffs[-1, 0]

    sigma = slip_input(
        t,
        sigma_0=sigma_0,
        omega=omega,
        amplitude_ratio=amplitude_ratio,
    )
    mu = stribeck_mu(sigma, params)
    phi, psi = structural_coefficients(params)
    alpha = jnp.abs(sigma) / mu * params.k_0

    transport_p0 = C.scalar_mul(dzdx_p0, -1.0 / params.L)
    coupled_mean_p0 = constant_p0(operators, psi * integral_z)
    relaxation_state_p0 = C.sub(z_p0, coupled_mean_p0)
    relaxation_p0 = C.scalar_mul(relaxation_state_p0, -alpha)
    trailing_edge_p0 = constant_p0(operators, (psi / params.L) * zL)
    slip_drive_p0 = constant_p0(operators, phi * sigma)

    rhs_p0 = C.add(
        C.add(transport_p0, relaxation_p0),
        C.add(trailing_edge_p0, slip_drive_p0),
    )
    return rhs_p0.coeffs.reshape(-1)


@partial(jax.jit, static_argnums=1)
def _simulate_dctkit_fixed_grid(
    params: TirePDEParameters,
    operators: DctkitLineOperators,
    xi: jax.Array,
    t: jax.Array,
    z0: jax.Array,
    sigma_0: ScalarLike,
    omega: ScalarLike,
    amplitude_ratio: ScalarLike,
) -> SimulationResult:
    dt = t[1] - t[0]

    def rhs(
        t_current: jax.Array,
        z_current: jax.Array,
        args: RhsArgs,
    ) -> jax.Array:
        params, operators, sigma_0, omega, amplitude_ratio = args
        return tire_pde_rhs_dctkit(
            t_current,
            z_current,
            params,
            operators,
            sigma_0=sigma_0,
            omega=omega,
            amplitude_ratio=amplitude_ratio,
        )

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        solver=diffrax.Tsit5(),
        t0=t[0],
        t1=t[-1],
        dt0=dt,
        y0=z0,
        args=(params, operators, sigma_0, omega, amplitude_ratio),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=100_000,
    )

    z = solution.ys
    force = compute_force_dctkit(z, params, operators)
    sigma = slip_input(
        t,
        sigma_0=sigma_0,
        omega=omega,
        amplitude_ratio=amplitude_ratio,
    )
    return SimulationResult(t=t, xi=xi, z=z, force=force, slip=sigma)


def simulate_pde_dctkit(
    params: TirePDEParameters = TirePDEParameters(),
    n_x: int = 100,
    t_span: tuple[float, float] = (0.0, 5.0),
    n_t: int = 5001,
    sigma_0: ScalarLike = 0.12,
    omega: ScalarLike = 5.0 * jnp.pi,
    amplitude_ratio: ScalarLike = 0.6,
    z0: ArrayLike | None = None,
) -> SimulationResult:
    """Run the dynamic tire PDE using dctkit cochain spatial operators."""

    if n_x < 2:
        raise ValueError("n_x must be at least 2.")
    if n_t < 2:
        raise ValueError("n_t must be at least 2.")

    operators = build_line_operators(n_x)
    dt = (t_span[1] - t_span[0]) / (n_t - 1)
    max_dt = 2.0 * float(params.L) * operators.dx
    if dt > max_dt:
        raise ValueError(
            f"Time step {dt:.6g} is too large for the explicit Diffrax solver; "
            f"use n_t >= {int((t_span[1] - t_span[0]) / max_dt) + 2}."
        )

    xi = jnp.linspace(0.0, 1.0, n_x)
    t = jnp.linspace(t_span[0], t_span[1], n_t)

    if z0 is None:
        z0 = jnp.zeros((n_x,))
    else:
        z0 = jnp.asarray(z0)

    return _simulate_dctkit_fixed_grid(
        params,
        operators,
        xi,
        t,
        z0,
        sigma_0,
        omega,
        amplitude_ratio,
    )


def compare_with_baseline(
    result: SimulationResult,
    baseline: SimulationResult,
) -> dict[str, float]:
    return {
        "max_abs_z_error": float(jnp.max(jnp.abs(result.z - baseline.z))),
        "max_abs_force_error": float(jnp.max(jnp.abs(result.force - baseline.force))),
        "max_abs_slip_error": float(jnp.max(jnp.abs(result.slip - baseline.slip))),
        "final_force_error": float(jnp.abs(result.force[-1] - baseline.force[-1])),
    }


def plot_dctkit_comparison(
    result: SimulationResult,
    baseline: SimulationResult,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Plot the DCTKit and standard PDE force histories on the same grid."""

    import matplotlib.pyplot as plt

    t = jax.device_get(result.t)
    dctkit_force = jax.device_get(result.force)
    baseline_force = jax.device_get(baseline.force)
    force_error = dctkit_force - baseline_force

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_force, ax_error) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 5.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    ax_force.plot(t, baseline_force, label="standard PDE", linewidth=1.4)
    ax_force.plot(
        t,
        dctkit_force,
        label="DCTKit cochains",
        linewidth=1.0,
        linestyle="--",
    )
    ax_force.set_ylabel(r"Tire force $F_x$ (N)")
    ax_force.grid(True)
    ax_force.legend()

    ax_error.plot(t, force_error, color="tab:red", linewidth=1.0)
    ax_error.axhline(0.0, color="black", linewidth=0.8)
    ax_error.set_xlabel(r"Travelled distance $s$ (m)")
    ax_error.set_ylabel(r"$\Delta F_x$ (N)")
    ax_error.grid(True)

    if output_dir is not None:
        fig.savefig(
            output_dir / "dctkit_standard_pde_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dynamic tire PDE using dctkit cochain operators."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved plot PNGs. Use an empty string to skip saving.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot windows after the simulation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run the simulation without creating plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = simulate_pde_dctkit()
    baseline = simulate_pde()
    errors = compare_with_baseline(result, baseline)

    print(f"Simulated {result.t.size} time steps and {result.xi.size} spatial points.")
    print(f"Final force: {float(result.force[-1]):.6g} N")
    print(f"Final slip: {float(result.slip[-1]):.6g}")
    print("Comparison with simulate_pde.py:")
    for key, value in errors.items():
        print(f"  {key}: {value:.6g}")

    if args.no_plots:
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    plot_simulation(result, output_dir=output_dir, show=args.show)
    plot_dctkit_comparison(result, baseline, output_dir=output_dir, show=args.show)
    if output_dir is not None:
        print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
