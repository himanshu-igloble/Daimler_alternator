"""Phase 2A: is GED an ordered continuum or a status enum?"""
import importlib.util, pathlib, json
import polars as pl
_SRC = pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")

def _spearman_np(x, y):
    """Fallback: Pearson on ranks (numpy only)."""
    import numpy as np
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx)**2).sum() * ((ry - my)**2).sum())
    rho = float(num / den) if den != 0 else 0.0
    # two-tailed p-value via t-distribution approximation
    import math
    t_stat = rho * math.sqrt(n - 2) / math.sqrt(max(1e-15, 1 - rho**2))
    # approximate p using normal for large n, else return nan
    try:
        from scipy.stats import t as t_dist
        p = float(2 * t_dist.sf(abs(t_stat), df=n-2))
    except Exception:
        p = float("nan")
    return rho, p

def main():
    df = pl.read_parquet(C.DAILY_CACHE)
    dom = df.with_columns(
        pl.when((pl.col("ged_cnt_3")>=pl.col("ged_cnt_2")) & (pl.col("ged_cnt_3")>pl.col("ged_cnt_0"))).then(3)
        .when(pl.col("ged_cnt_2")>0).then(2).otherwise(0).alias("dom_state")
    ).filter(pl.col("vsi_mean").is_not_null())
    med = dom.group_by("dom_state").agg(pl.col("vsi_mean").median().alias("vsi_med"),
                                        pl.len().alias("n")).sort("dom_state")
    dom_states = dom["dom_state"].to_list()
    vsi_vals   = dom["vsi_mean"].to_list()
    # Try scipy first, fall back to numpy
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(dom_states, vsi_vals)
        rho, p = float(rho), float(p)
    except Exception:
        rho, p = _spearman_np(dom_states, vsi_vals)
    verdict = ("ENUM (non-monotonic / state-3 is a CAN-fault cluster)"
               if abs(rho) < 0.3 else "ORDINAL signal warrants further test")
    out = {"per_state_median_vsi": med.to_dicts(), "spearman_rho": rho, "spearman_p": p, "verdict": verdict}
    (C.RESULTS/"2a_ordinality.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
