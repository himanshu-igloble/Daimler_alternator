"""Phase 3B: VSI regulation-effort proxy (feasible substitute for excitation reconstruction).

3A (infeasibility) is documented in the Phase-5 report, not here: the dataset lacks
alternator output current, rotor/field current, and alternator temperature, so true
continuous excitation E(t) cannot be regressed. This module instead models the healthy-fleet
conditional surface E[VSI | RPM, ANR, CSP] and treats the residual as a regulation-effort proxy.

Theory
------
A healthy alternator self-regulates to maintain system voltage near ~28 V under all load
and speed conditions. The regulator drives rotor-field current up when RPM is low or
electrical load is high, and backs off at cruise. The observable proxy is:

    resid(t) = VSI(t) - E[VSI | RPM(t), ANR(t), CSP(t)]_healthy-fleet

where the reference surface is estimated non-parametrically (median per operating-point bin)
from the 15 non-failed trucks' engine-on data.

Interpretation
--------------
- resid ~ 0        : voltage matches healthy-fleet expectation for that operating point
- resid << 0       : voltage chronically below what healthy trucks achieve at same conditions
                     => degraded regulation or increased series resistance
- resid trending down over last 30 days => deteriorating regulation effort
- resid_neg_frac   : fraction of engine-on rows with sub-expected voltage (stressor severity)
- resid_oscillation: std-dev of daily mean residuals (healthy trucks are stable)
"""
import importlib.util
import pathlib
import sys

import polars as pl
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent


def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = _load("ged_common")

# Bin widths for the (RPM, ANR, CSP) operating-point surface
RPM_BIN = 100.0   # rev/min
ANR_BIN = 100.0   # Nm
CSP_BIN = 10.0    # km/h


def _bins(df: pl.DataFrame) -> pl.DataFrame:
    """Attach integer bin keys (rb, ab, cb) derived from RPM, ANR, CSP."""
    return df.with_columns([
        (pl.col("RPM") // RPM_BIN).alias("rb"),
        (pl.col("ANR") // ANR_BIN).alias("ab"),
        (pl.col("CSP") // CSP_BIN).alias("cb"),
    ])


def fit_reference(nf_eo: pl.DataFrame) -> pl.DataFrame:
    """Estimate the healthy-fleet conditional median surface E[VSI | RPM, ANR, CSP].

    Parameters
    ----------
    nf_eo : pl.DataFrame
        Engine-on rows from non-failed trucks. Must contain columns
        RPM, ANR, CSP, VSI (no nulls required — dropped internally).

    Returns
    -------
    pl.DataFrame with columns: rb, ab, cb, vsi_exp
        One row per occupied operating-point bin.
    """
    d = _bins(nf_eo.drop_nulls(["RPM", "ANR", "CSP", "VSI"]))
    return (
        d.group_by(["rb", "ab", "cb"])
        .agg(pl.col("VSI").median().alias("vsi_exp"))
    )


def residual(df: pl.DataFrame, surf: pl.DataFrame) -> pl.DataFrame:
    """Compute per-row VSI residuals against the healthy-fleet reference surface.

    Rows whose operating-point bin has no reference coverage are dropped (inner join).

    Parameters
    ----------
    df   : pl.DataFrame  — VIN data (engine-on or full; nulls dropped internally)
    surf : pl.DataFrame  — output of fit_reference()

    Returns
    -------
    pl.DataFrame with all input columns plus `resid` = VSI - vsi_exp,
    restricted to rows with matching reference bins.
    """
    d = _bins(df.drop_nulls(["RPM", "ANR", "CSP", "VSI"])).join(
        surf, on=["rb", "ab", "cb"], how="inner"
    )
    return d.with_columns(
        (pl.col("VSI") - pl.col("vsi_exp")).alias("resid")
    )


def engine_on(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only engine-on rows: RPM > 0 and not null."""
    return df.filter((pl.col("RPM") > 0) & pl.col("RPM").is_not_null())


def _sample_eo(vin: str, cap: int = 500_000, seed: int = 0) -> pl.DataFrame:
    """Load engine-on rows for one VIN, sampling down to `cap` rows if needed.

    This bounds memory when concatenating all 15 NF VINs for surface fitting.
    Without the cap the NF fleet is ~60M engine-on rows; with it at most 7.5M.
    """
    eo = engine_on(C.load_vin(vin)).drop_nulls(["RPM", "ANR", "CSP", "VSI"])
    if eo.height > cap:
        return eo.sample(cap, seed=seed)
    return eo


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import time

    print("Building reference surface from non-failed VINs …")
    t0 = time.time()
    nf_parts = []
    for v in C.NONFAILED_VINS:
        part = _sample_eo(v)
        nf_parts.append(part)
        print(f"  {v}: {part.height:,} engine-on rows (capped at 500k)")
    nf = pl.concat(nf_parts)
    print(f"  Reference pool: {nf.height:,} rows total")
    surf = fit_reference(nf)
    print(f"  Surface bins: {surf.height:,}  (elapsed {time.time()-t0:.1f}s)")

    print("\nComputing per-VIN residual features …")
    rows = []
    for vin in C.ALL_VINS:
        t1 = time.time()
        vin_df = engine_on(C.load_vin(vin))
        r = residual(vin_df, surf)
        if r.height == 0:
            print(f"  {vin}: no matching bins — skipped")
            continue
        daily = (
            r.group_by("day")
            .agg(pl.col("resid").mean().alias("rd"))
            .sort("day")
        )
        x = np.arange(daily.height)
        y = daily["rd"].to_numpy()
        last = min(30, daily.height)
        slope = float(np.polyfit(x[-last:], y[-last:], 1)[0]) if last >= 2 else float("nan")
        rows.append({
            "vin": vin,
            "failed": C.is_failed(vin),
            "resid_mean": float(np.nanmean(r["resid"].to_numpy())),
            "resid_neg_frac": float((r["resid"] < 0).mean()),
            "resid_oscillation": float(np.nanstd(y)),
            "resid_slope_30d": float(slope),
        })
        elapsed = time.time() - t1
        print(f"  {vin}: n={r.height:,}  mean={rows[-1]['resid_mean']:+.3f}  slope30={slope:+.4f}  ({elapsed:.1f}s)")

    out = pl.DataFrame(rows)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.RESULTS / "3b_regulation_features.csv")
    print(f"\nSaved -> {C.RESULTS / '3b_regulation_features.csv'}")
    print("\n--- Per-VIN regulation-effort proxy features (sorted by resid_mean) ---")
    print(out.sort("resid_mean"))


if __name__ == "__main__":
    main()
