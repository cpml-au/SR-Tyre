from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_and_process_bins(data_type=1, n_bins=5, n_points=200):
    if data_type == 0:
        df = pd.read_csv(DATA_DIR / "longitudinal_tire_test.csv")
    else:
        df = pd.read_csv(DATA_DIR / "lateral_tire_test.csv")

    SR = df["SR"].to_numpy()
    SA = df["SA"].to_numpy()
    FX = df["FX"].to_numpy()
    FY = df["FY"].to_numpy()
    FZ = df["FZ"].to_numpy()

    slip_ratio = SR
    sigma_x = slip_ratio / (1 + slip_ratio) + 0.05
    sigma_y = np.tan(SA * np.pi / 180 / (1 + slip_ratio))

    Fx = FX
    Fy = -FY
    Fz = -FZ * 4.448  # N

    if data_type == 0:
        F = Fx
        sigma = sigma_x
    else:
        F = Fy
        sigma = sigma_y

    Fz_edges = np.linspace(Fz.min(), Fz.max(), n_bins + 1)
    Fz_bin = np.digitize(Fz, Fz_edges)  # 1..n_bins+1

    sigma_cells = []
    F_cells = []

    for i in range(1, n_bins + 1):
        mask = Fz_bin == i

        s = sigma[mask]
        f = F[mask]

        if len(s) == 0:
            sigma_cells.append(np.array([]))
            F_cells.append(np.array([]))
            continue

        order = np.argsort(s)
        s = s[order]
        f = f[order]

        valid = np.isfinite(s) & np.isfinite(f)
        s, f = s[valid], f[valid]

        mask_slip = np.abs(s) > 1e-4
        s, f = s[mask_slip], f[mask_slip]

        med = np.median(f)
        mad = np.median(np.abs(f - med)) + 1e-12
        mask_mad = np.abs(f - med) < 2 * mad
        s, f = s[mask_mad], f[mask_mad]

        if len(s) > 0:
            unique_s, inv = np.unique(s, return_inverse=True)
            unique_f = np.array([f[inv == k].mean() for k in range(len(unique_s))])
        else:
            unique_s, unique_f = np.array([]), np.array([])

        sigma_cells.append(unique_s)
        F_cells.append(unique_f)

    Fz_rep = np.array([
        np.mean(Fz[Fz_bin == i]) if np.any(Fz_bin == i) else np.nan
        for i in range(1, n_bins + 1)
    ])

    y_data = [F_cells[i] / Fz_rep[i] for i in range(n_bins)]
    x_data = sigma_cells

    x_new, y_new = [], []

    for i in range(n_bins):
        if len(x_data[i]) < 2:
            x_new.append(np.array([]))
            y_new.append(np.array([]))
            continue

        x_i = np.linspace(x_data[i].min(), x_data[i].max(), n_points)
        interp = PchipInterpolator(x_data[i], y_data[i])
        y_i = interp(x_i)

        x_new.append(x_i)
        y_new.append(y_i)

    return x_new, y_new, Fz_rep


def make_datasets(data_type=1, n_bins=5, n_points=200):
    x_bins, y_bins, _ = load_and_process_bins(
        data_type=data_type,
        n_bins=n_bins,
        n_points=n_points,
    )

    # Use bins 1-3 for training, bin 4 for validation, and bin 5 for testing.
    train_x_bins = x_bins[: min(3, len(x_bins))]
    train_y_bins = y_bins[: min(3, len(y_bins))]
    val_x_bins = x_bins[3:4]
    val_y_bins = y_bins[3:4]
    test_x_bins = x_bins[4:5]
    test_y_bins = y_bins[4:5]

    X_train = np.concatenate(train_x_bins)
    y_train = np.concatenate(train_y_bins)
    X_val = np.concatenate(val_x_bins)
    y_val = np.concatenate(val_y_bins)
    X_test = np.concatenate(test_x_bins)
    y_test = np.concatenate(test_y_bins)

    return X_train, y_train, X_val, y_val, X_test, y_test


def reshape_flattened_bins(values, Fz_rep):
    values = np.asarray(values, dtype=float).reshape(-1)
    Fz_rep = np.atleast_1d(np.asarray(Fz_rep, dtype=float)).reshape(-1)

    if values.size % Fz_rep.size != 0:
        raise ValueError(
            f"Cannot reshape {values.size} values into {Fz_rep.size} load bins."
        )

    return values.reshape(Fz_rep.size, -1)
