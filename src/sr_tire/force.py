from functools import lru_cache

import numpy as np

DEFAULT_THETA_OPT = np.array([
    0.0668, 0.0001, 360.4850, 0.0230,
    0.6456, 3.07e-05,
    1.7213, 3.6888, 1.3652,
])


def default_mu_expression(v, mu_s, v_s, delta_s):
    return 1 + (mu_s - 1) * np.exp(-np.abs(v / v_s) ** delta_s)


@lru_cache(maxsize=None)
def _get_force_model_grids(V, n_v, n_x, epsilon):
    v = np.linspace(-1, 1, n_v) * V
    xi_bar = np.linspace(0, 1, n_x)
    v_abs = np.sqrt(v**2 + epsilon)
    v_sign = v / v_abs
    return v, xi_bar, v_abs, v_sign


def compute_force_model(
    Fz_rep,
    theta_opt=None,
    V=16,
    n_v=200,
    n_x=100,
    epsilon=1e-12,
    mu_expression=None,
):
    Fz_rep = np.asarray(Fz_rep, dtype=float).reshape(-1)

    if theta_opt is None:
        theta_opt = DEFAULT_THETA_OPT

    if mu_expression is None:
        mu_expression = default_mu_expression

    L = theta_opt[0] + theta_opt[1] * np.sqrt(Fz_rep)
    p = Fz_rep / L

    mu_d = theta_opt[4] - theta_opt[5] * Fz_rep
    k0 = (theta_opt[2] - theta_opt[3] * Fz_rep) / mu_d
    mu_s = theta_opt[6]
    v_s = theta_opt[7]
    delta_s = theta_opt[8]

    v, xi_bar, v_abs, v_sign = _get_force_model_grids(V, n_v, n_x, epsilon)
    F_b = np.zeros((len(Fz_rep), n_v))
    mu = np.asarray(mu_expression(v, mu_s, v_s, delta_s), dtype=float).reshape(-1)
    if mu.size == 1:
        mu = np.full(n_v, mu.item(), dtype=float)
    elif mu.size != n_v:
        raise ValueError(
            f"mu_expression must return either a scalar or an array of length {n_v}, "
            f"got shape {mu.shape}"
        )

    for k in range(len(Fz_rep)):
        xi = xi_bar * L[k]
        dx = xi[1] - xi[0]
        base = -(mu / k0[k]) * v_sign
        exponent = -(v_abs[:, None] / V) * (k0[k] / mu)[:, None] * xi[None, :]
        z_sum = np.sum(base[:, None] * (1 - np.exp(exponent)), axis=1)
        F_b[k, :] = z_sum * dx * k0[k] * p[k] * mu_d[k]

    return v, F_b
