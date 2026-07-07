import argparse
from pathlib import Path
from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "pde"


class TirePDEParameters(NamedTuple):
    """Physical and friction parameters for the tire PDE model.

    Attributes:
        L: Contact patch length in meters.
        k_0: Normalized bristle stiffness in 1 / meters.
        Fz: Vertical load on the tire in newtons.
        w: Carcass stiffness in newtons / meter.
        mu_s: Static friction coefficient.
        mu_d: Dynamic friction coefficient.
        v_S: Stribeck velocity scale in meters / second.
        delta_S: Stribeck curve exponent.
        V: Forward rolling speed in meters / second.
    """

    # Structural/contact parameters.
    L: float = 0.082
    k_0: float = 732.0
    Fz: float = 3000.0
    w: float = 4.5e5

    # Friction-law parameters.
    mu_s: float = 1.3
    mu_d: float = 0.84
    v_S: float = 5.24
    delta_S: float = 0.86
    V: float = 16.0


class SimulationResult(NamedTuple):
    """Arrays returned by the simulation.

    Attributes:
        t: Travelled-distance grid used as the integration variable.
        xi: Dimensionless spatial grid along the contact patch, from 0 to 1.
        z: Bristle deflection field with shape ``(n_t, n_x)``.
        force: Longitudinal tire force over the travelled-distance grid.
        slip: Longitudinal slip input over the travelled-distance grid.
    """

    t: jax.Array
    xi: jax.Array
    z: jax.Array
    force: jax.Array
    slip: jax.Array


def structural_coefficients(params):
    """Return carcass/bristle coupling coefficients."""

    phi = params.w / (params.k_0 * params.Fz + params.w)
    psi = 1.0 - phi
    return phi, psi


def slip_input(t, sigma_0=0.12, omega=5.0 * jnp.pi, amplitude_ratio=0.6):
    """Sinusoidal longitudinal slip input.

    Args:
        t: Travelled distance variable.
        sigma_0: Mean longitudinal slip.
        omega: Spatial excitation frequency in 1 / meters.
        amplitude_ratio: Oscillation amplitude as a fraction of ``sigma_0``.
    """

    return sigma_0 + amplitude_ratio * sigma_0 * jnp.sin(omega * t)


def stribeck_mu(sigma, params):
    """Stribeck friction coefficient as a function of slip."""

    v_slip = jnp.abs(params.V * sigma)
    return params.mu_d + (params.mu_s - params.mu_d) * jnp.exp(
        -((v_slip / params.v_S) ** params.delta_S)
    )


def _expand_to_state(value, z):
    while jnp.ndim(value) < jnp.ndim(z):
        value = value[..., jnp.newaxis]
    return value


def tire_pde_rhs_from_sigma(z, sigma, params, dx):
    """Semi-discrete PDE right-hand side for an already-sampled slip value."""

    mu = stribeck_mu(sigma, params)
    phi, psi = structural_coefficients(params)

    # alpha controls how strongly bristle deflection relaxes toward the
    # friction-limited state for the current slip and friction coefficient.
    alpha = jnp.abs(sigma) / mu * params.k_0

    # Upwind spatial derivative with z(xi=0) = 0 imposed at the leading edge.
    z_left = jnp.concatenate((jnp.zeros_like(z[..., :1]), z[..., :-1]), axis=-1)
    dzdx = (z - z_left) / dx

    # Integral and trailing-edge values couple the bristles through the carcass.
    integral_z = jnp.trapezoid(z, dx=dx, axis=-1)
    zL = z[..., -1]

    alpha = _expand_to_state(alpha, z)
    integral_z = _expand_to_state(integral_z, z)
    zL = _expand_to_state(zL, z)
    sigma = _expand_to_state(sigma, z)

    return (
        -(1.0 / params.L) * dzdx
        - alpha * (z - psi * integral_z)
        + (psi / params.L) * zL
        + phi * sigma
    )


def tire_pde_rhs(
    t,
    z,
    params,
    dx,
    sigma_0=0.12,
    omega=5.0 * jnp.pi,
    amplitude_ratio=0.6,
):
    """Semi-discrete dynamic tire PDE right-hand side.

    Args:
        t: Current travelled distance.
        z: Bristle deflection values on the spatial grid.
        params: Physical tire and friction parameters.
        dx: Spacing of the dimensionless spatial grid ``xi``.
        sigma_0: Mean slip used by ``slip_input``.
        omega: Slip excitation frequency used by ``slip_input``.
        amplitude_ratio: Slip oscillation amplitude as a fraction of ``sigma_0``.
    """

    sigma = slip_input(
        t,
        sigma_0=sigma_0,
        omega=omega,
        amplitude_ratio=amplitude_ratio,
    )
    return tire_pde_rhs_from_sigma(z, sigma, params, dx)


def compute_force(z, xi, params):
    """Integrate bristle deflection to obtain longitudinal tire force."""

    return params.Fz * params.k_0 * jnp.trapezoid(z, x=xi, axis=-1)


@jax.jit
def _simulate_fixed_grid(params, xi, t, z0, sigma_0, omega, amplitude_ratio):
    dx = xi[1] - xi[0]
    dt = t[1] - t[0]

    def rhs(t_current, z_current, args):
        params, dx, sigma_0, omega, amplitude_ratio = args
        return tire_pde_rhs(
            t_current,
            z_current,
            params,
            dx,
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
        args=(params, dx, sigma_0, omega, amplitude_ratio),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=100_000,
    )

    z = solution.ys
    force = compute_force(z, xi, params)
    sigma = slip_input(
        t,
        sigma_0=sigma_0,
        omega=omega,
        amplitude_ratio=amplitude_ratio,
    )
    return SimulationResult(t=t, xi=xi, z=z, force=force, slip=sigma)


def _interpolate_uniform_history(t_current, t_grid, values):
    """Linearly interpolate rows of a uniformly sampled slip history."""

    dt = t_grid[1] - t_grid[0]
    raw_index = (t_current - t_grid[0]) / dt
    left_index = jnp.floor(raw_index).astype(jnp.int32)
    left_index = jnp.clip(left_index, 0, t_grid.size - 2)
    fraction = raw_index - left_index
    fraction = jnp.clip(fraction, 0.0, 1.0)
    left = values[..., left_index]
    right = values[..., left_index + 1]
    return left + fraction * (right - left)


@jax.jit
def _simulate_sampled_slip(params, xi, t, sigma, z0):
    dx = xi[1] - xi[0]
    dt = t[1] - t[0]

    def rhs(t_current, z_current, args):
        params, dx, t_grid, sigma_history = args
        sigma_current = _interpolate_uniform_history(t_current, t_grid, sigma_history)
        return tire_pde_rhs_from_sigma(z_current, sigma_current, params, dx)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        solver=diffrax.Tsit5(),
        t0=t[0],
        t1=t[-1],
        dt0=dt,
        y0=z0,
        args=(params, dx, t, sigma),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=100_000,
    )

    z = jnp.moveaxis(solution.ys, 0, 1)
    force = compute_force(z, xi, params)
    return SimulationResult(t=t, xi=xi, z=z, force=force, slip=sigma)


def simulate_pde(
    params=TirePDEParameters(),
    n_x=100,
    t_span=(0.0, 5.0),
    n_t=5001,
    sigma_0=0.12,
    omega=5.0 * jnp.pi,
    amplitude_ratio=0.6,
    z0=None,
):
    """Run the dynamic tire PDE model on a fixed output grid.

    This implementation uses Diffrax's ``Tsit5`` ODE solver and saves the
    solution on a fixed output grid, so the rollout remains JIT-compilable and
    differentiable with respect to JAX-compatible inputs.

    Args:
        params: Physical tire and friction parameters.
        n_x: Number of spatial grid points in the contact patch.
        t_span: Start and end of the travelled-distance interval.
        n_t: Number of saved solution points in ``t_span``.
        sigma_0: Mean longitudinal slip.
        omega: Spatial excitation frequency in 1 / meters.
        amplitude_ratio: Slip oscillation amplitude as a fraction of ``sigma_0``.
        z0: Optional initial bristle deflection array of shape ``(n_x,)``.
    """

    if n_x < 2:
        raise ValueError("n_x must be at least 2.")
    if n_t < 2:
        raise ValueError("n_t must be at least 2.")

    dx = 1.0 / (n_x - 1)
    dt = (t_span[1] - t_span[0]) / (n_t - 1)
    max_dt = 2.0 * float(params.L) * dx
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

    return _simulate_fixed_grid(params, xi, t, z0, sigma_0, omega, amplitude_ratio)


def simulate_pde_for_slip_history(
    t,
    sigma,
    params=TirePDEParameters(),
    n_x=100,
    z0=None,
):
    """Run the PDE model using sampled slip input.

    Args:
        t: Uniform travelled-distance grid with shape ``(n_t,)``.
        sigma: Slip history with shape ``(n_t,)`` or ``(n_runs, n_t)``.
        params: Physical tire and friction parameters.
        n_x: Number of spatial grid points in the contact patch.
        z0: Optional initial deflection with shape ``(n_x,)`` or
            ``(n_runs, n_x)``.
    """

    t_np = np.asarray(t, dtype=float)
    sigma_np = np.asarray(sigma, dtype=float)
    scalar_history = sigma_np.ndim == 1
    if scalar_history:
        sigma_np = sigma_np[np.newaxis, :]

    if n_x < 2:
        raise ValueError("n_x must be at least 2.")
    if t_np.ndim != 1 or t_np.size < 2:
        raise ValueError("t must be a one-dimensional array with at least 2 points.")
    if sigma_np.shape[-1] != t_np.size:
        raise ValueError("The last dimension of sigma must match the size of t.")

    dt_values = np.diff(t_np)
    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError("Sampled-slip simulation expects a uniformly sampled t grid.")

    dx = 1.0 / (n_x - 1)
    dt = float((t_np[-1] - t_np[0]) / (t_np.size - 1))
    max_dt = 2.0 * float(params.L) * dx
    if dt > max_dt:
        raise ValueError(
            f"Time step {dt:.6g} is too large for the explicit Diffrax solver; "
            f"use a finer t grid or n_x <= {int(2.0 * float(params.L) / dt) + 1}."
        )

    xi = jnp.linspace(0.0, 1.0, n_x)
    t = jnp.asarray(t_np)
    sigma = jnp.asarray(sigma_np)
    if z0 is None:
        z0 = jnp.zeros((sigma.shape[0], n_x))
    else:
        z0 = jnp.asarray(z0)
        if z0.ndim == 1:
            z0 = jnp.broadcast_to(z0, (sigma.shape[0], n_x))

    result = _simulate_sampled_slip(params, xi, t, sigma, z0)
    if scalar_history:
        return SimulationResult(
            t=result.t,
            xi=result.xi,
            z=result.z[0],
            force=result.force[0],
            slip=result.slip[0],
        )
    return result


def plot_simulation(result, output_dir=None, show=True):
    """Create bristle deformation, force, and slip figures."""

    import matplotlib.pyplot as plt

    xi = jax.device_get(result.xi)
    t = jax.device_get(result.t)
    z = jax.device_get(result.z)
    force = jax.device_get(result.force)
    slip = jax.device_get(result.slip)

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    fig, ax = plt.subplots()
    ax.plot(xi, z[-1], linewidth=1)
    ax.set_xlabel(r"Longitudinal coordinate $\xi$ (-)")
    ax.set_ylabel(r"Bristle deflection $z$ (m)")
    ax.grid(True)
    figures.append(("bristle_deformation.png", fig))

    fig, ax = plt.subplots()
    ax.plot(t, force, linewidth=1)
    ax.set_xlabel(r"Travelled distance $s$ (m)")
    ax.set_ylabel(r"Tire force $F_x$ (N)")
    ax.grid(True)
    figures.append(("force.png", fig))

    fig, ax = plt.subplots()
    ax.plot(t, slip, linewidth=1)
    ax.set_xlabel(r"Travelled distance $s$ (m)")
    ax.set_ylabel(r"Longitudinal slip $\sigma$ (-)")
    ax.grid(True)
    figures.append(("slip.png", fig))

    if output_dir is not None:
        for filename, fig in figures:
            fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        for _, fig in figures:
            plt.close(fig)

    return [fig for _, fig in figures]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the dynamic tire PDE simulation.")
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


def main():
    args = parse_args()
    result = simulate_pde()
    print(f"Simulated {result.t.size} time steps and {result.xi.size} spatial points.")
    print(f"Final force: {float(result.force[-1]):.6g} N")
    print(f"Final slip: {float(result.slip[-1]):.6g}")

    if args.no_plots:
        return

    output_dir = args.output_dir or None
    plot_simulation(result, output_dir=output_dir, show=args.show)
    if output_dir is not None:
        print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
