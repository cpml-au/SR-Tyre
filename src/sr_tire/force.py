import numpy as np

DEFAULT_THETA_OPT = np.array([
    0.0668, 0.0001, 360.4850, 0.0230,
    0.6456, 3.07e-05,
    1.7213, 3.6888, 1.3652,
])


def default_mu_expression(v, mu_s, v_s, delta_s):
    return 1 + (mu_s - 1) * np.exp(-np.abs(v / v_s) ** delta_s)


def compute_force_model(
    Fz_rep,
    theta_opt=None,
    V=16,
    n_v=200,
    n_x=100,
    epsilon=1e-12,
    mu_expression=None,
):
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

    v = np.linspace(-1, 1, n_v) * V
    xi_bar = np.linspace(0, 1, n_x)
    F_b = np.zeros((len(Fz_rep), n_v))

    for k in range(len(Fz_rep)):
        xi = xi_bar * L[k]
        dx = xi[1] - xi[0]
        mu = np.asarray(mu_expression(v, mu_s, v_s, delta_s), dtype=float).reshape(-1)
        if mu.size == 1:
            mu = np.full(n_v, mu.item(), dtype=float)
        elif mu.size != n_v:
            raise ValueError(
                f"mu_expression must return either a scalar or an array of length {n_v}, "
                f"got shape {mu.shape}"
            )
        z = np.zeros((n_v, n_x))

        for i in range(n_v):
            for j in range(n_x):
                z[i, j] = (
                    -mu[i] / k0[k]
                    * v[i]
                    / np.sqrt(v[i] ** 2 + epsilon)
                    * (
                        1
                        - np.exp(
                            -np.sqrt(v[i] ** 2 + epsilon) / V * k0[k] / mu[i] * xi[j]
                        )
                    )
                )

        F_b[k, :] = np.sum(z, axis=1) * dx * k0[k] * p[k] * mu_d[k]

    return v, F_b
