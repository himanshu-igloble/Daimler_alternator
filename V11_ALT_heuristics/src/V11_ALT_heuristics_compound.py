"""V11_ALT_heuristics — #11 compound weak-vote early-watch alarm.

Fires "early-watch" at the earliest horizon where >= VOTE_MIN of the orthogonal
VOTE_CHANNELS cross their NF p05/p95 boundary in the bad direction (weak vote: no
within-truck z requirement). The GED==2 storm remains a separate high-precision
emergency and is NOT part of the vote. Evaluated over the 10 failed VINs and,
for false-alarm honesty, the 15 NF trucks (LOO envelope).
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
FOR = _load("V11_ALT_heuristics_forensic")
FORENSICS = cfg.FORENSICS


def count_votes(win_means: pd.Series, nfb: pd.DataFrame, cfg) -> int:
    """Count vote channels whose window mean crosses the NF boundary the bad way."""
    n = 0
    for f in cfg.VOTE_CHANNELS:
        v = win_means.get(f, np.nan)
        if pd.isna(v):
            continue
        p05, p95 = nfb.loc[f, "nf_p05"], nfb.loc[f, "nf_p95"]
        if f in cfg.BAD_HIGH:
            n += int(v > p95)
        else:
            n += int(v < p05)
    return n


def first_trigger(d_all: pd.DataFrame, nfb: pd.DataFrame, cfg):
    """Earliest horizon label where >= VOTE_MIN channels vote; returns (label, n)."""
    d = d_all[d_all["n_eo"] >= cfg.MIN_EO_SAMPLES]
    for lo, hi, lbl in cfg.HORIZON_BINS:
        win = d[(d["dtf"] > lo) & (d["dtf"] <= hi)]
        if win.empty:
            continue
        means = win[cfg.VOTE_CHANNELS].mean()
        nv = count_votes(means, nfb, cfg)
        if nv >= cfg.VOTE_MIN:
            return lbl, nv
    return None, 0


def main() -> None:
    nfb = pd.read_csv(FORENSICS / "nf_baseline.csv").set_index("feature")
    nf_vins = [v for v in cfg.ALL_VINS if v not in cfg.FAILED_VIN_SET]

    rows = []
    for vin in cfg.FAILED_VIN_SET:
        d = pd.read_csv(FORENSICS / f"{vin}_daily.csv")
        lbl, nv = first_trigger(d, nfb, cfg)
        rows.append({"group": "FAILED", "vin_label": vin,
                     "early_watch_horizon_days": (lbl if lbl else "none"),
                     "n_votes": nv, "fired": bool(lbl)})

    nf_dailies = {v: pd.read_csv(FORENSICS / f"{v}_daily.csv") for v in nf_vins}
    for vin in nf_vins:
        loo = [nf_dailies[v] for v in nf_vins if v != vin]
        nfb_loo = FOR._nf_baseline(loo).set_index("feature")
        lbl, nv = first_trigger(nf_dailies[vin], nfb_loo, cfg)
        rows.append({"group": "NF", "vin_label": vin,
                     "early_watch_horizon_days": (lbl if lbl else "none"),
                     "n_votes": nv, "fired": bool(lbl)})

    out = pd.DataFrame(rows)
    out.to_csv(FORENSICS / "compound_alarm_lovo.csv", index=False)
    fired_f = int(out[(out.group == "FAILED") & out.fired].shape[0])
    fired_nf = int(out[(out.group == "NF") & out.fired].shape[0])
    print(f"[v11 compound] early-watch recall: {fired_f}/10 failed; "
          f"false alarms: {fired_nf}/15 NF")
    leads = out[(out.group == "FAILED") & out.fired]["early_watch_horizon_days"].tolist()
    print(f"  first-trigger lead horizons (failed): {leads}")


if __name__ == "__main__":
    main()
