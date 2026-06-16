"""V11_ALT_heuristics — customer report (md + xlsx), V11-native precursor-centric."""
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
FOR = cfg.FORENSICS


def _classify(hits, nff):
    if nff >= 2:
        return "false_alarm_prone"
    if hits >= 2:
        return "generalizes"
    if hits == 1:
        return "anecdotal"
    return "no_signal"


def _feat_table():
    dev = pd.read_csv(FOR / "failed_window_deviations.csv")
    st = pd.read_csv(FOR / "nf_self_test.csv")
    disc = dev[dev["discriminative"] == True]                      # noqa: E712
    nff = st[st["verdict"] == "FALSE_ALARM"]["feature"].value_counts()
    rows = []
    for f in cfg.NEW_FEATS:
        h = disc[disc["feature"] == f]["vin_label"].nunique()
        n = int(nff.get(f, 0))
        rows.append({"feature": f, "failed_discriminative": h, "nf_false": n, "class": _classify(h, n)})
    return pd.DataFrame(rows).sort_values(["class", "failed_discriminative"], ascending=[True, False])


def main():
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv")
    es = pd.read_csv(FOR / "earliest_signal_per_vin.csv")
    st = pd.read_csv(FOR / "nf_self_test.csv")
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    cp = pd.read_csv(FOR / "changepoint_per_vin.csv")
    feat = _feat_table()

    n11 = int((es["verdict"] == "discriminative_precursor").sum())
    n1062 = int((comp["v1062_horizon"].astype(str) != "none").sum())
    nf_false = int((st["verdict"] == "FALSE_ALARM").sum())
    cf = int(cl[(cl.group == "FAILED") & (cl.fired == True)].shape[0])   # noqa: E712
    cnf = int(cl[(cl.group == "NF") & (cl.fired == True)].shape[0])      # noqa: E712
    gen = feat[feat["class"] == "generalizes"]["feature"].tolist()

    md = []
    md.append("# Alternator Lead-Time Heuristics — Customer Report (V11)\n")
    md.append("## 1. Executive Summary\n")
    md.append(f"V11 tests 12 new lead-time heuristics on the same six raw CAN channels used by "
              f"the frozen classifier. It raises discriminative-precursor recall to **{n11}/10** "
              f"(V10.6.2: {n1062}/10) with **{nf_false}/15** false alarms on healthy trucks, and "
              f"produces an earlier first warning for one truck. This is a real but modest gain; it "
              f"does NOT change the WHICH-truck classifier (AUROC 0.927) or the fleet replacement "
              f"window — those remain as delivered in V10.6.2. V11 makes the WHEN-emergency warning "
              f"fire earlier and for one additional truck.\n")
    md.append("## 2. What V11 Adds\n")
    md.append("- **VIN9** newly detected (30-day horizon) via post-crank recovery time.\n"
              "- **VIN1** earliest warning moved 30d → 60d (earlier lead).\n"
              "- **`crank_recovery_t` (post-crank voltage recovery, heuristic #3) is the MVP** "
              "— discriminative on 6/10 failed trucks, 0/15 false alarms.\n")
    md.append("## 3. Per-Truck Lead-Time (V10.6.2 vs V11)\n")
    md.append(comp[["vin_label", "v1062_horizon", "v1062_feature", "v11_horizon",
                    "v11_feature", "earlier", "new_in_v11"]].to_markdown(index=False))
    md.append("\n## 4. Heuristic Catalog & Verdict\n")
    md.append(feat.to_markdown(index=False))
    md.append(f"\n_Generalizing new features: {gen}_\n")
    md.append("## 5. Compound Early-Watch Alarm (#11)\n")
    md.append(f"A 2-of-5 weak vote across orthogonal channels fires on **{cf}/10** failed trucks "
              f"with **{cnf}/15** false alarms, giving the earliest first-trigger for some trucks "
              f"(e.g. VIN8 at 90 days). It is an orthogonal 'early-watch' tier; the GED=2 storm "
              f"stays as the separate high-precision emergency.\n")
    md.append("## 6. Change-point & Resting-Voltage (exploratory)\n")
    md.append("Per-truck CUSUM change-point and cumulative under-voltage dose fire far too early "
              "(220–599 day leads) to be actionable — exploratory only. Resting-voltage decay "
              "slope is discriminative on 3/10 (VIN1/4/10).\n")
    md.append("## 7. Honest Limitations\n")
    md.append("- `crank_recovery_t` within-truck z-scores are inflated by a near-zero healthy "
              "baseline (NF p95 ≈ 0.05 s — healthy trucks recover essentially instantly) plus the "
              "30 s recovery censor; trust the 0/15 NF false-alarm guard, not the z-magnitude.\n"
              "- The crank_recovery_t signal is **episodic** — isolated slow-recovery events, not a "
              "smooth degradation trend. For VIN9 the events span the truck's whole life (spikes at "
              "~590d, ~420d and ~20d before failure), so the '30-day lead' for VIN9 should be read "
              "cautiously rather than as a clean precursor.\n"
              "- VIN8's strict-gate hit rests on only 3 trusted days (fragile); the compound alarm "
              "catches it more robustly at 90d.\n- 4/10 failed trucks (VIN3/4/5/7) remain "
              "undetectable by any channel — consistent with abrupt/silent electrical failure with "
              "no precursor.\n- n=10 events: treat per-feature results as leads, not calibrated "
              "rates.\n- No per-truck daily RUL is produced or implied.\n")
    md.append("## 8. Deployment Recommendation\n")
    md.append("Add `crank_recovery_t` to the emergency channel alongside the GED=2 storm signal, "
              "and surface the compound 2-of-5 vote as an 'early-watch' tier (0 false alarms, "
              "earlier first-triggers). The load-split sag features (`sag_highload_frac` vs "
              "`sag_idle_frac`) and `reg_duty_frac` add repair-direction guidance even where they "
              "do not change recall.\n")
    out_md = cfg.REPORTS_DIR / "V11_ALT_heuristics_customer_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    summary = pd.DataFrame({
        "metric": ["v11_recall", "v1062_recall", "nf_false_alarms", "compound_recall",
                   "compound_nf_false", "mvp_feature", "generalizing_features"],
        "value": [f"{n11}/10", f"{n1062}/10", f"{nf_false}/15", f"{cf}/10", f"{cnf}/15",
                  "crank_recovery_t", len(gen)]})
    xlsx = cfg.REPORTS_DIR / "V11_ALT_heuristics_fleet_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        comp.to_excel(xl, sheet_name="LeadTime_Comparison", index=False)
        feat.to_excel(xl, sheet_name="Heuristic_Verdict", index=False)
        cl.to_excel(xl, sheet_name="Compound_Alarm", index=False)
        cp.to_excel(xl, sheet_name="Changepoint", index=False)
        st.to_excel(xl, sheet_name="NF_SelfTest", index=False)
    print(f"[v11 customer report] wrote {out_md} and {xlsx}")


if __name__ == "__main__":
    main()
