"""V11_ALT_heuristics — pure daily feature functions (no file IO, no globals).

Each function takes a prepared per-VIN DataFrame (see prepare()) or a subset and
returns a pandas Series/DataFrame indexed by 'day'. Kept pure so the 12 heuristics
are unit-testable on tiny synthetic frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    vsi = df["VSI"].where(df["VSI"] <= 36, df["VSI"] * 0.2)
    df["vsi"] = vsi.where((vsi > 10) & (vsi <= 36))
    df["eo"] = (df["RPM"] > 0) & (df["RPM"] <= 3500)
    df["off"] = (df["RPM"] == 0)
    df["anr"] = df["ANR"].where((df["ANR"] >= cfg.ANR_VALID_LO) & (df["ANR"] <= cfg.ANR_VALID_HI))
    df["t_s"] = pd.to_datetime(df["DATETIME"]).astype("int64").to_numpy() / 1e9
    df["day"] = df["DAYS_SINCE_SALE"]
    return df


def _ols_slope(x, y, min_n: int = 10, min_span: float = 50.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < min_n or np.ptp(x) < min_span:
        return np.nan
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / sxx)


def reg_duty(eo: pd.DataFrame, cfg) -> pd.Series:
    """#4 fraction of engine-on samples with VSI in [REG_BAND_LO, REG_BAND_HI]."""
    inb = eo["vsi"].between(cfg.REG_BAND_LO, cfg.REG_BAND_HI)
    return inb.groupby(eo["day"]).mean().rename("reg_duty_frac")


def crank_effort(df: pd.DataFrame, eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#7 cranks per engine-hour and mean crank (SMA==1 run-length) duration."""
    rise = (df["SMA"] == 1) & (df["SMA"].shift(1) == 0)
    starts = rise.groupby(df["day"]).sum()
    ehrs = eo.groupby("day").size() * cfg.SAMPLE_SEC / 3600.0
    cph = (starts / ehrs).replace([np.inf, -np.inf], np.nan).rename("cranks_per_ehr")
    on = (df["SMA"] == 1).to_numpy().astype(int)
    dayv = df["day"].to_numpy()
    runs = []
    i, n = 0, len(on)
    while i < n:
        if on[i] == 1:
            j = i
            while j < n and on[j] == 1:
                j += 1
            runs.append((dayv[i], (j - i) * cfg.SAMPLE_SEC))
            i = j
        else:
            i += 1
    if runs:
        rr = pd.DataFrame(runs, columns=["day", "dur"])
        dur = rr.groupby("day")["dur"].mean()
    else:
        dur = pd.Series(dtype=float)
    return pd.DataFrame({"cranks_per_ehr": cph, "crank_dur_mean": dur})


def ged_states(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """#8 GED==1 and GED==3 daily rates, plus 0->2 / 0->3 transition churn."""
    day = df["day"]
    g1 = (df["GED"] == 1).groupby(day).mean()
    g3 = (df["GED"] == 3).groupby(day).mean()
    churn = ((df["GED"].shift(1) == 0) & (df["GED"].isin([2, 3]))).groupby(day).sum()
    return pd.DataFrame({"ged1_frac": g1, "ged3_frac": g3, "ged_churn": churn})


def vsi_rpm_curve(eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#1 charging-ramp slope dVSI/dRPM (600-1500 rpm), regulation ceiling
    (mean VSI at RPM>1500), and charging-onset rpm (first RPM reaching VSI_TARGET)."""
    ramp = eo[(eo["RPM"] >= cfg.RAMP_RPM_LO) & (eo["RPM"] <= cfg.RAMP_RPM_HI)]
    slope = ramp.groupby("day").apply(
        lambda g: _ols_slope(g["RPM"].to_numpy(), g["vsi"].to_numpy()))
    ceiling = eo[eo["RPM"] > cfg.CEILING_RPM_MIN].groupby("day")["vsi"].mean()

    def _onset(g):
        hit = g.loc[g["vsi"] >= cfg.VSI_TARGET, "RPM"]
        return float(hit.min()) if len(hit) else np.nan

    onset = eo.groupby("day").apply(_onset)
    out = pd.DataFrame({"vsi_rpm_slope": slope, "vsi_ceiling": ceiling,
                        "vsi_onset_rpm": onset})
    return out


def _bin3(d: pd.DataFrame, cfg) -> pd.DataFrame:
    d = d.dropna(subset=["vsi", "anr", "RPM", "CSP"]).copy()
    d["rpm_bin"] = np.floor(d["RPM"] / cfg.REF_RPM_BIN).astype("int64")
    d["anr_bin"] = np.floor(d["anr"] / cfg.REF_ANR_BIN).astype("int64")
    d["csp_bin"] = np.floor(d["CSP"] / cfg.REF_CSP_BIN).astype("int64")
    return d


def build_load_reference(nf_eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#2 binned healthy-fleet surface E[VSI | RPM, ANR, CSP] (median per bin,
    only bins with >= REF_MIN_BIN_COUNT NF samples). Index = (rpm_bin,anr_bin,csp_bin)."""
    d = _bin3(nf_eo, cfg)
    g = d.groupby(["rpm_bin", "anr_bin", "csp_bin"])["vsi"]
    ref = pd.DataFrame({"vsi_med": g.median(), "n": g.size()})
    return ref[ref["n"] >= cfg.REF_MIN_BIN_COUNT]


def load_residual(eo: pd.DataFrame, ref: pd.DataFrame, cfg) -> pd.DataFrame:
    """#2 daily residual mean (obs - expected at actual operating point) and
    negative-residual fraction. Samples whose bin is absent from ref are dropped."""
    d = _bin3(eo, cfg)
    d = d.join(ref["vsi_med"], on=["rpm_bin", "anr_bin", "csp_bin"]).dropna(subset=["vsi_med"])
    if d.empty:
        return pd.DataFrame({"vsi_resid_mean": [], "vsi_resid_negfrac": []})
    d["resid"] = d["vsi"] - d["vsi_med"]
    rmean = d.groupby("day")["resid"].mean()
    rneg = (d["resid"] < 0).groupby(d["day"]).mean()
    return pd.DataFrame({"vsi_resid_mean": rmean, "vsi_resid_negfrac": rneg})


def crank_recovery(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """#3 per SMA 1->0 edge: seconds for VSI to reach VSI_TARGET (censored at
    RECOVERY_WINDOW_S if never reached) and the recovery slope (V/s) over that
    window; returns per-day mean across start events."""
    s = df.sort_values("t_s")
    sma = s["SMA"].to_numpy()
    vsi = s["vsi"].to_numpy()
    t = s["t_s"].to_numpy()
    dayv = s["day"].to_numpy()
    prev = np.r_[0, sma[:-1]]
    fall = np.where((prev == 1) & (sma == 0))[0]
    rows = []
    W = cfg.RECOVERY_WINDOW_S
    for i in fall:
        t0 = t[i]
        j = i
        reached = np.nan
        xs, ys = [], []
        while j < len(t) and (t[j] - t0) <= W:
            v = vsi[j]
            if not np.isnan(v):
                xs.append(t[j] - t0)
                ys.append(v)
                if v >= cfg.VSI_TARGET and np.isnan(reached):
                    reached = t[j] - t0
            j += 1
        rt = reached if not np.isnan(reached) else (W if xs else np.nan)
        sl = _ols_slope(xs, ys, min_n=3, min_span=5.0)
        rows.append((dayv[i], rt, sl))
    if not rows:
        return pd.DataFrame({"crank_recovery_t": [], "crank_recovery_slope": []})
    r = pd.DataFrame(rows, columns=["day", "rt", "sl"])
    return pd.DataFrame({"crank_recovery_t": r.groupby("day")["rt"].mean(),
                         "crank_recovery_slope": r.groupby("day")["sl"].mean()})


def idle_hunting(eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#9 within-idle (RPM 550-950, CSP<5) VSI variance, lag-1 autocorr, and
    mean-crossing rate (regulator-hunting / ripple signature)."""
    idle = eo[(eo["RPM"].between(cfg.IDLE_RPM_LO, cfg.IDLE_RPM_HI))
              & (eo["CSP"] < cfg.IDLE_CSP_MAX)].sort_values("t_s")

    def _acf1(v):
        v = v.dropna().to_numpy()
        if len(v) < 10 or v.std(ddof=0) == 0:
            return np.nan
        return float(np.corrcoef(v[:-1], v[1:])[0, 1])

    def _zcr(v):
        v = v.dropna().to_numpy()
        if len(v) < 10:
            return np.nan
        c = np.sign(v - v.mean())
        c = c[c != 0]
        if len(c) < 2:
            return np.nan
        return float(np.mean(np.diff(c) != 0))

    var = idle.groupby("day")["vsi"].var()
    acf = idle.groupby("day")["vsi"].apply(_acf1)
    zcr = idle.groupby("day")["vsi"].apply(_zcr)
    return pd.DataFrame({"idle_vsi_var": var, "idle_vsi_acf1": acf, "idle_vsi_zcr": zcr})


def sag_typing(eo: pd.DataFrame, cfg) -> pd.DataFrame:
    """#10 under-voltage (VSI<SAG_V) fraction split by load: high-torque
    (ANR>HIGH_LOAD_NM, stator/diode) vs low-load (ANR<LOW_LOAD_NM, regulator)."""
    hi = eo[eo["anr"] > cfg.HIGH_LOAD_NM]
    lo = eo[eo["anr"] < cfg.LOW_LOAD_NM]
    hf = (hi["vsi"] < cfg.SAG_V).groupby(hi["day"]).mean()
    lf = (lo["vsi"] < cfg.SAG_V).groupby(lo["day"]).mean()
    return pd.DataFrame({"sag_highload_frac": hf, "sag_idle_frac": lf})


def uv_dose_daily(eo: pd.DataFrame, cfg) -> pd.Series:
    """#5 daily under-voltage dose increment = integral of (SAG_V - VSI)*dt while
    engine-on and VSI<SAG_V (volt-seconds), using real DATETIME deltas (gap-capped)."""
    d = eo.sort_values("t_s").copy()
    d["dt"] = d.groupby("day")["t_s"].diff().clip(upper=cfg.DOSE_DT_CAP_S).fillna(0.0)
    under = d["vsi"] < cfg.SAG_V
    d["dose"] = np.where(under & d["vsi"].notna(), (cfg.SAG_V - d["vsi"]) * d["dt"], 0.0)
    return d.groupby("day")["dose"].sum().rename("uv_dose_day")
