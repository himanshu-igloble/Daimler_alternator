"""V11.1_ALT — 3-channel emergency layer.
Channel 1 GED=2 storm (daily count >= GED_EMERGENCY_DAILY_COUNT_MIN).
Channel 2 crank-recovery exceedance (>= k trusted days > NF p95 within a
trailing EXCEED_TRAIL_DAYS window; k calibrated to 0/15 NF fires).
Channel 3 compound 2-of-5 vote (x2_compound evaluated per trusted day).
Firing rules never read dtf; leads (failed trucks) are post-hoc reporting only."""
from __future__ import annotations
import importlib.util, pathlib
import numpy as np, pandas as pd
_src = pathlib.Path(__file__).resolve().parent
def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_src / f"{n}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cfg = _load("V11_1_ALT_config")
C = _load("V11_1_ALT_covariates")

def exceedance_first_fire(panel, p95, k, trail):
    d = panel[panel["n_eo"] >= cfg.MIN_EO_SAMPLES].sort_values("day")
    days = d.loc[pd.to_numeric(d["crank_recovery_t"], errors="coerce") > p95, "day"].to_numpy()
    for i in range(len(days)):
        if i + 1 >= k and days[i] - days[i + 1 - k] <= trail:
            return int(days[i])
    return None

def ged_first_fire(panel):
    d = panel.sort_values("day")
    hit = d.loc[pd.to_numeric(d["ged2_cnt"], errors="coerce") >= cfg.GED_EMERGENCY_DAILY_COUNT_MIN, "day"]
    return int(hit.iloc[0]) if len(hit) else None

def compound_first_fire(panel, nfb):
    d = C.trusted(panel).sort_values("day")
    for t in d["day"].tolist():
        if C.x2_compound(panel, t, nfb) == 1:
            return int(t)
    return None

def _lead(panel, fire_day):
    if fire_day is None:
        return ""
    d = panel[panel["day"] == fire_day]
    if d.empty or pd.isna(d["dtf"].iloc[0]):
        return ""
    return round(float(d["dtf"].iloc[0]), 1)

def main() -> None:
    cfg.EMERG_CACHE.mkdir(parents=True, exist_ok=True)
    p95 = float((cfg.COV_CACHE / "crank_p95.txt").read_text())
    nfb = pd.read_csv(cfg.V11_FORENSICS / "nf_baseline.csv").set_index("feature")
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET)
    panels = {v: C.load_panel(v) for v in lc["vin_label"]}
    nf_vins = lc[lc.failed_flag == False]["vin_label"].tolist()      # noqa: E712
    f_vins = lc[lc.failed_flag == True]["vin_label"].tolist()        # noqa: E712

    # calibrate k: smallest k >= EXCEED_K_START with 0 NF fires
    k = cfg.EXCEED_K_START
    print("[emergency] exceedance k calibration:")
    while True:
        nf_fires = sum(exceedance_first_fire(panels[v], p95, k, cfg.EXCEED_TRAIL_DAYS) is not None
                       for v in nf_vins)
        f_fires = sum(exceedance_first_fire(panels[v], p95, k, cfg.EXCEED_TRAIL_DAYS) is not None
                      for v in f_vins)
        print(f"  k={k}: NF fires {nf_fires}/15, failed fires {f_fires}/10")
        if nf_fires == 0 or k >= 10:
            break
        k += 1

    rows = []
    for _, r in lc.iterrows():
        v = r["vin_label"]; panel = panels[v]
        g = ged_first_fire(panel)
        e = exceedance_first_fire(panel, p95, k, cfg.EXCEED_TRAIL_DAYS)
        c = compound_first_fire(panel, nfb)
        # CURRENT-STATE channels (decision-engine inputs): evaluated at the
        # truck's latest trusted day only — one evaluation, no whole-life
        # multiple-comparison inflation. Ever-fired columns above are honest
        # negative-result reporting, NOT deployable standing alarms.
        d_tr = C.trusted(panel)
        t_now = float(d_tr["day"].max()) if len(d_tr) else float("nan")
        comp_now = int(C.x2_compound(panel, t_now, nfb)) if len(d_tr) else 0
        recent = d_tr[(d_tr["day"] > t_now - cfg.EXCEED_TRAIL_DAYS) & (d_tr["day"] <= t_now)]
        exceed_now = int((pd.to_numeric(recent["crank_recovery_t"], errors="coerce")
                          > p95).sum() >= cfg.EXCEED_K_START) if len(d_tr) else 0
        rows.append({"vin_label": v, "failed_flag": int(r["failed_flag"]),
                     "ged_fired": g is not None, "ged_lead_days": _lead(panel, g),
                     "exceed_fired": e is not None, "exceed_lead_days": _lead(panel, e),
                     "exceed_k": k,
                     "compound_fired": c is not None, "compound_lead_days": _lead(panel, c),
                     "any_fired": any(x is not None for x in (g, e, c)),
                     "compound_current": comp_now, "exceed_current": exceed_now,
                     # early-watch = compound only: the exceedance channel failed
                     # every honest test (no viable k ever-fired; 1/15 NF current)
                     # and is reported but NOT decision-feeding.
                     "early_watch_current": comp_now})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.EMERG_CACHE / "emergency_per_vin.csv", index=False)
    f = out[out.failed_flag == 1]; nf = out[out.failed_flag == 0]
    print(f"\n[emergency] EVER-FIRED (whole-life, reporting only) failed recall: "
          f"ged {int(f.ged_fired.sum())}/10, exceed {int(f.exceed_fired.sum())}/10, "
          f"compound {int(f.compound_fired.sum())}/10")
    print(f"[emergency] EVER-FIRED NF false fires: ged {int(nf.ged_fired.sum())}, "
          f"exceed {int(nf.exceed_fired.sum())}, compound {int(nf.compound_fired.sum())} "
          f"(channels 2-3 NOT deployable as standing alarms)")
    print(f"[emergency] CURRENT-STATE (decision inputs): failed early-watch "
          f"{int(f.early_watch_current.sum())}/10; NF false {int(nf.early_watch_current.sum())}/15 "
          f"(gate target 0)")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
