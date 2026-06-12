"""V11.1_ALT — leakage-safe covariates from V11 daily panels.
x1(t) = log1p(#trusted days <= t with crank_recovery_t > NF p95)
x2(t) = 1 if >= VOTE_MIN of VOTE_CHANNELS cross the NF boundary (bad direction)
        on the trailing X2_TRAIL_DAYS age-window mean ending at t.
dtf is NEVER read here (post-hoc knowledge)."""
from __future__ import annotations
import importlib.util, pathlib
import numpy as np, pandas as pd
_src = pathlib.Path(__file__).resolve().parent
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_src / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cfg = _load("V11_1_ALT_config")

def trusted(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["n_eo"] >= cfg.MIN_EO_SAMPLES]

def x1_exceedance(panel: pd.DataFrame, t: float, p95: float) -> float:
    d = trusted(panel)
    n = int(((d["day"] <= t) & (pd.to_numeric(d["crank_recovery_t"], errors="coerce") > p95)).sum())
    return float(np.log1p(n))

def x2_compound(panel: pd.DataFrame, t: float, nfb: pd.DataFrame) -> int:
    d = trusted(panel)
    win = d[(d["day"] > t - cfg.X2_TRAIL_DAYS) & (d["day"] <= t)]
    if win.empty:
        return 0
    votes = 0
    for ch in cfg.VOTE_CHANNELS:
        v = pd.to_numeric(win[ch], errors="coerce").mean()
        if pd.isna(v):
            continue
        if ch in cfg.VOTE_BAD_HIGH:
            votes += int(v > nfb.loc[ch, "nf_p95"])
        else:
            votes += int(v > nfb.loc[ch, "nf_p95"])
    return int(votes >= cfg.VOTE_MIN)

def load_panel(vin: str) -> pd.DataFrame:
    return pd.read_csv(cfg.V11_FORENSICS / f"{vin}_daily.csv")

def crank_p95_from_nf(nf_vins) -> float:
    """NF p95 of crank_recovery_t pooled over the given NF trucks' trusted days."""
    vals = []
    for v in nf_vins:
        d = trusted(load_panel(v))
        vals.append(pd.to_numeric(d["crank_recovery_t"], errors="coerce").dropna())
    return float(pd.concat(vals).quantile(0.95))

def covariate_vector(vin: str, t: float, p95: float, nfb: pd.DataFrame):
    panel = load_panel(vin)
    return x1_exceedance(panel, t, p95), x2_compound(panel, t, nfb)

def main() -> None:
    """Stage output: whole-fleet covariates at each truck's endpoint (fit-time x)."""
    cfg.COV_CACHE.mkdir(parents=True, exist_ok=True)
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    nf_vins = [v for v in lc[lc.failed_flag == False]["vin_label"]]          # noqa: E712
    p95 = crank_p95_from_nf(nf_vins)
    nfb = pd.read_csv(cfg.V11_FORENSICS / "nf_baseline.csv").set_index("feature")
    rows = []
    for _, r in lc.iterrows():
        t_end = float(r["age_days_observed"])
        x1, x2 = covariate_vector(r["vin_label"], t_end, p95, nfb)
        rows.append({"vin_label": r["vin_label"], "failed_flag": int(r["failed_flag"]),
                     "t_end": t_end, "x1": round(x1, 4), "x2": x2})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.COV_CACHE / "covariates_fit.csv", index=False)
    (cfg.COV_CACHE / "crank_p95.txt").write_text(str(p95), encoding="utf-8")
    print(f"[covariates] p95={p95:.4f}; fit-time covariates:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
