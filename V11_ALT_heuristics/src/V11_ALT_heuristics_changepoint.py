"""V11_ALT_heuristics — per-truck change-point lead-time module.

#12 CUSUM change-point on each truck's own standardized feature series (the
change-point timestamp IS the lead time). #5 knee detection on the cumulative
under-voltage dose curve. #6 life-long resting-voltage decay slope vs the NF
slope envelope. Reads the daily panels written by the forensic stage.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FORENSICS = cfg.FORENSICS


def cusum_changepoint(x, direction="down", k=0.5, h=5.0, block=5):
    """Tabular CUSUM on block-averaged z-standardized x. Averages 'block'
    consecutive samples before standardizing to suppress high-frequency
    oscillations (periodic noise). Returns the index (in original sample
    space) where the one-sided cumulative sum first exceeds h, backtracked
    to the start of the accumulation run, else None."""
    x = np.asarray(x, dtype=float)
    m = ~np.isnan(x)
    if m.sum() < 20:
        return None
    x_valid = x[m]
    n_blocks = len(x_valid) // block
    if n_blocks < 4:
        return None
    blocks = x_valid[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    mu = np.mean(blocks[: n_blocks // 3])
    sd = np.std(blocks[: n_blocks // 3])
    if sd <= 0:
        sd = np.std(blocks)
    if sd <= 0:
        return None
    z = (blocks - mu) / sd
    s = 0.0
    start_block = None
    for i, zi in enumerate(z):
        if direction == "down":
            prev_s = s
            s = max(0.0, s - zi - k)
            if s > 0 and prev_s == 0:
                start_block = i
            if s == 0:
                start_block = None
            if s > h:
                return (start_block if start_block is not None else i) * block
        else:
            prev_s = s
            s = max(0.0, s + zi - k)
            if s > 0 and prev_s == 0:
                start_block = i
            if s == 0:
                start_block = None
            if s > h:
                return (start_block if start_block is not None else i) * block
    return None


def knee_index(cum):
    """Kneedle-style elbow: index of max distance from the chord joining the
    first and last points of the (monotone) cumulative curve."""
    y = np.asarray(cum, dtype=float)
    n = len(y)
    if n < 5 or y[-1] == y[0]:
        return None
    x = np.arange(n)
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(np.argmax(d))


def _lead_at(d, idx):
    """dtf at panel row idx = lead time in days (None if idx invalid)."""
    if idx is None or idx < 0 or idx >= len(d):
        return None
    val = d["dtf"].iloc[idx]
    return None if pd.isna(val) else float(val)


def _resting_slope(d):
    x = d["day"].to_numpy(dtype=float)
    y = d["resting_vsi_mean"].to_numpy(dtype=float)
    m = ~np.isnan(y)
    if m.sum() < 15:
        return np.nan
    xm, ym = x[m].mean(), y[m].mean()
    sxx = ((x[m] - xm) ** 2).sum()
    return float(((x[m] - xm) * (y[m] - ym)).sum() / sxx) if sxx > 0 else np.nan


def main() -> None:
    nf_slopes = []
    for vin in cfg.ALL_VINS:
        if vin in cfg.FAILED_VIN_SET:
            continue
        d = pd.read_csv(FORENSICS / f"{vin}_daily.csv").sort_values("day")
        d = d[d["n_eo"] >= cfg.MIN_EO_SAMPLES]
        nf_slopes.append(_resting_slope(d))
    nf_slopes = [s for s in nf_slopes if not np.isnan(s)]
    rest_p05 = float(np.quantile(nf_slopes, 0.05)) if nf_slopes else np.nan

    rows = []
    for vin in cfg.FAILED_VIN_SET:
        d = pd.read_csv(FORENSICS / f"{vin}_daily.csv").sort_values("day")
        d = d[d["n_eo"] >= cfg.MIN_EO_SAMPLES].reset_index(drop=True)
        cp_resid = cusum_changepoint(d["vsi_resid_mean"].to_numpy(), "down",
                                     cfg.CUSUM_K, cfg.CUSUM_H)
        cp_rest = cusum_changepoint(d["resting_vsi_mean"].to_numpy(), "down",
                                    cfg.CUSUM_K, cfg.CUSUM_H)
        cum = np.nancumsum(d["uv_dose_day"].to_numpy())
        knee = knee_index(cum) if np.nanmax(cum) > 0 else None
        slope = _resting_slope(d)
        rows.append({
            "vin_label": vin,
            "cp_resid_lead_days": _lead_at(d, cp_resid),
            "cp_resting_lead_days": _lead_at(d, cp_rest),
            "dose_knee_lead_days": _lead_at(d, knee),
            "resting_slope": round(slope, 5) if not np.isnan(slope) else "",
            "resting_slope_nf_p05": round(rest_p05, 5) if not np.isnan(rest_p05) else "",
            "resting_slope_disc": bool(not np.isnan(slope) and not np.isnan(rest_p05)
                                       and slope < rest_p05),
        })
    out = pd.DataFrame(rows)
    out.to_csv(FORENSICS / "changepoint_per_vin.csv", index=False)
    print("[v11 changepoint] per-VIN change-point / knee / resting-slope:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
