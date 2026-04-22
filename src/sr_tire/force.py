from functools import lru_cache

import numpy as np

MODEL_V = 16

# fix with paper values
DEFAULT_THETA_OPT = np.array([
    0.0668, 0.0001, 360.4850, 0.0230,
    0.6456, 3.07e-05,
    1.7213, 3.6888, 1.3652,
])


def stribeck(v, mu_s, v_s, delta_s):
    return 1 + (mu_s - 1) * np.exp(-np.abs(v / v_s) ** delta_s)


@lru_cache(maxsize=None)
def _get_force_model_grids(V, n_v, epsilon):
    v = np.linspace(-1, 1, n_v) * V
    v_abs = np.sqrt(v**2 + epsilon)
    v_sign = np.sign(v)
    return v, v_abs, v_sign


def compute_force_model(
    Fz_rep,
    theta_opt=None,
    V=MODEL_V,
    n_v=200,
    epsilon=1e-12,
    mu_expression=None,
    v=None,
):
    if theta_opt is None:
        theta_opt = DEFAULT_THETA_OPT

    if mu_expression is None:
        mu_expression = stribeck
        # mu_s = mu_s_bar
        mu_s = theta_opt[6]
        v_s = theta_opt[7]
        delta_s = theta_opt[8]

    # L = a_1 + a_2 * sqrt(Fz); Fz is bin-dependent representative load
    L = theta_opt[0] + theta_opt[1] * np.sqrt(Fz_rep)

    # mu_d = theta = a_5 - a_6 * Fz
    mu_d = theta_opt[4] - theta_opt[5] * Fz_rep
    # k0 = k0_paper/theta = (a_3 - a_4 * Fz) / mu_d
    k0 = (theta_opt[2] - theta_opt[3] * Fz_rep) / mu_d

    if v is None:
        v, v_abs, v_sign = _get_force_model_grids(V, n_v, epsilon)
    else:
        v_abs = np.sqrt(v**2 + epsilon)
        v_sign = np.sign(v)
        n_v = v.shape[-1]

    mu = mu_expression(v, mu_s, v_s, delta_s)

    a = k0[:, None] * v_abs / (V * mu)
    aL = a * L[:, None]
    F_b = (
        -mu
        * ((np.exp(-aL) - 1) / aL + 1)
        * Fz_rep[:, None]
        * v_sign
        * mu_d[:, None]
    )

    return v, F_b
