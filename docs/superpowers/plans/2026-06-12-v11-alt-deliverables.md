# V11 Deliverables Implementation Plan (graphs → report → decks)

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Build V11-native graphs, customer report (md+xlsx), and two pptx decks — V11 data only, mirroring V10.6.2 artifact types and house style.

**Architecture:** Three standalone generator modules under `V11_ALT_heuristics/` that read existing V11 caches and write images / md / xlsx / pptx. No pipeline re-runs. matplotlib Agg, python-pptx, openpyxl (via pandas).

**Tech Stack:** Python 3.11 `py -3`; matplotlib, pandas, numpy, python-pptx, openpyxl. Verify availability in Task 1.

**Run dir:** repo root `D:\Daimler-starter_motor_alternator_battery`. Scripts run from `V11_ALT_heuristics\src` or `presentation`.

**Verified data contracts (do not re-derive):** see spec §1
`docs/superpowers/specs/2026-06-12-v11-alt-deliverables-design.md`. Key:
comparison.csv (vin_label,v1062_horizon,v1062_feature,v11_horizon,v11_feature,earlier,new_in_v11);
failed_window_deviations.csv (…feature,discriminative True/False);
nf_self_test.csv (…feature,verdict FALSE_ALARM/clean);
compound_alarm_lovo.csv (group,vin_label,early_watch_horizon_days,n_votes,fired);
changepoint_per_vin.csv; nf_baseline.csv (feature,nf_p05,nf_p50,nf_p95,…);
<VIN>_daily.csv (day,n_eo,…crank_recovery_t,dtf,vin_label).
Verified headline: V11 6/10, V10.6.2 5/10, NF 0/15, compound 4/10, MVP crank_recovery_t.
Horizons are strings "90/60/45/30/14/7" or "none"→0.

---

## Task 1: Toolchain check + graphs module skeleton + G1/G3

**Files:**
- Create: `V11_ALT_heuristics/src/V11_ALT_heuristics_graphs.py`
- Output dir: `V11_ALT_heuristics/visualizations/V11_graphs/`

- [ ] **Step 1: Verify libraries**

Run: `py -3 -c "import matplotlib, pptx, openpyxl, pandas; print('mpl',matplotlib.__version__,'pptx',pptx.__version__,'openpyxl',openpyxl.__version__)"`
Expected: prints versions. If `pptx` or `openpyxl` ImportError → `py -3 -m pip install python-pptx openpyxl` then retry. (Task 4/5 need pptx; report needs openpyxl.)

- [ ] **Step 2: Create the module with config loader, style, helpers, G1, G3, and a `main()` that calls only G1+G3 for now**

```python
"""V11_ALT_heuristics — graph suite (V11-native lead-time deliverables).

Reads committed V11 caches and writes professional PNG (+ HD) graphs mirroring
the V10.6.2 house style. No pipeline re-runs. matplotlib Agg backend.
"""
from __future__ import annotations

import importlib.util
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT = cfg.V11_ROOT / "visualizations" / "V11_graphs"

NAVY, GOLD, GREEN, RED, BLUE, GREY = "#0D1B2A", "#C58B1F", "#27AE60", "#C0392B", "#2980B9", "#95A5A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": NAVY,
                     "axes.labelcolor": NAVY, "text.color": NAVY,
                     "xtick.color": NAVY, "ytick.color": NAVY, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 110})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}_hd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def _hzn(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def g1_recall_comparison():
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv")
    n11 = int((comp["v11_horizon"].astype(str) != "none").sum())
    n1062 = int((comp["v1062_horizon"].astype(str) != "none").sum())
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    comp_fired = int(cl[(cl.group == "FAILED") & (cl.fired == True)].shape[0])  # noqa: E712
    st = pd.read_csv(FOR / "nf_self_test.csv")
    nf_false = int((st["verdict"] == "FALSE_ALARM").sum())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ["V10.6.2\nstrict gate", "V11\nstrict gate", "V11\ncompound (#11)"]
    vals = [n1062, n11, comp_fired]
    colors = [GREY, GREEN, BLUE]
    b = ax.bar(bars, vals, color=colors, width=0.6)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, f"{v}/10",
                ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Failed trucks with a precursor (of 10)")
    ax.set_title("V11 lead-time recall vs V10.6.2  —  NF false alarms: "
                 f"{nf_false}/15", fontweight="bold", color=NAVY)
    ax.text(0.5, -0.18, "Honest gate: within-truck z>=2 AND outside healthy-fleet p05-p95",
            transform=ax.transAxes, ha="center", fontsize=9, color=GREY)
    return _save(fig, "G1_recall_comparison")


def g3_leadtime_dumbbell():
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv")
    comp = comp.copy()
    comp["a"] = comp["v1062_horizon"].map(_hzn)
    comp["b"] = comp["v11_horizon"].map(_hzn)
    comp = comp.sort_values("b")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(comp))
    for i, (_, r) in enumerate(comp.iterrows()):
        ax.plot([r["a"], r["b"]], [i, i], color=GREY, zorder=1, lw=2)
        ax.scatter(r["a"], i, color=GREY, s=70, zorder=2, label="V10.6.2" if i == 0 else "")
        col = GOLD if r["new_in_v11"] else (GREEN if r["earlier"] else BLUE)
        ax.scatter(r["b"], i, color=col, s=90, zorder=3, label="V11" if i == 0 else "")
    ax.set_yticks(y)
    ax.set_yticklabels(comp["vin_label"])
    ax.set_xlabel("Earliest discriminative horizon (days before failure)")
    ax.set_title("Per-truck earliest precursor: V10.6.2 → V11\n"
                 "(gold = newly detected, green = earlier lead)", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, "G3_leadtime_dumbbell")


def main():
    written = [g1_recall_comparison(), g3_leadtime_dumbbell()]
    print("[v11 graphs] wrote:", written)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it** — `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_graphs.py"`
Expected: prints `wrote: ['G1_recall_comparison.png', 'G3_leadtime_dumbbell.png']`; files exist in `visualizations/V11_graphs/` (PNG + _hd.png each). Open/stat to confirm non-zero size.

- [ ] **Step 4: Commit**
```
git add V11_ALT_heuristics/src/V11_ALT_heuristics_graphs.py V11_ALT_heuristics/visualizations/V11_graphs
git commit -m "feat(v11-alt): graphs G1 recall + G3 lead-time dumbbell"
```

---

## Task 2: G2 feature generalization + G4 compound + G5 crank-recovery + G6 changepoint + generation report

**Files:** Modify `V11_ALT_heuristics/src/V11_ALT_heuristics_graphs.py` (append 4 functions + report writer; extend `main()`).

- [ ] **Step 1: Append these functions**

```python
def _feat_class(failed_hits, nf_false):
    if nf_false >= 2:
        return "false_alarm_prone", RED
    if failed_hits >= 2:
        return "generalizes", GREEN
    if failed_hits == 1:
        return "anecdotal", GOLD
    return "no_signal", GREY


def g2_feature_generalization():
    dev = pd.read_csv(FOR / "failed_window_deviations.csv")
    st = pd.read_csv(FOR / "nf_self_test.csv")
    disc = dev[dev["discriminative"] == True]                      # noqa: E712
    nf_false_by = st[st["verdict"] == "FALSE_ALARM"]["feature"].value_counts()
    rows = []
    for f in cfg.NEW_FEATS:
        hits = disc[disc["feature"] == f]["vin_label"].nunique()
        nff = int(nf_false_by.get(f, 0))
        cls, col = _feat_class(hits, nff)
        rows.append((f, hits, nff, cls, col))
    d = pd.DataFrame(rows, columns=["feature", "hits", "nf_false", "cls", "col"]).sort_values("hits")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(d["feature"], d["hits"], color=d["col"])
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["hits"] + 0.05, i, str(int(r["hits"])), va="center", fontsize=9)
        if r["feature"] == "crank_recovery_t":
            ax.text(r["hits"] + 0.5, i, "  ← MVP (#3 post-crank recovery)",
                    va="center", color=RED, fontweight="bold", fontsize=10)
    ax.set_xlabel("Failed trucks discriminative (of 10)")
    ax.set_title("New-heuristic generalization (19 V11 features)\n"
                 "green=generalizes(≥2)  gold=anecdotal(1)  grey=no signal", fontweight="bold")
    ax.set_xlim(0, max(7, d["hits"].max() + 2))
    return _save(fig, "G2_feature_generalization")


def g4_compound_alarm_leads():
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    f = cl[(cl.group == "FAILED") & (cl.fired == True)].copy()      # noqa: E712
    f["h"] = f["early_watch_horizon_days"].map(_hzn)
    f = f.sort_values("h")
    nf_false = int(cl[(cl.group == "NF") & (cl.fired == True)].shape[0])  # noqa: E712
    fig, ax = plt.subplots(figsize=(9, 5))
    b = ax.barh(f["vin_label"], f["h"], color=BLUE)
    for rect, (_, r) in zip(b, f.iterrows()):
        ax.text(r["h"] + 0.7, rect.get_y() + rect.get_height() / 2,
                f"{int(r['h'])}d  ({int(r['n_votes'])} votes)", va="center", fontsize=10)
    ax.set_xlabel("First-trigger lead (days before failure)")
    ax.set_title(f"Compound 2-of-5 early-watch alarm (#11)  —  NF false alarms: {nf_false}/15",
                 fontweight="bold")
    ax.set_xlim(0, max(100, f["h"].max() + 25))
    return _save(fig, "G4_compound_alarm_leads")


def g5_crank_recovery_trajectories():
    nfb = pd.read_csv(FOR / "nf_baseline.csv").set_index("feature")
    p95 = float(nfb.loc["crank_recovery_t", "nf_p95"])
    vins = ["VIN1_F_ALT", "VIN8_F_ALT", "VIN9_F_ALT"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vin in zip(axes, vins):
        d = pd.read_csv(FOR / f"{vin}_daily.csv")
        d = d[(d["n_eo"] >= cfg.MIN_EO_SAMPLES) & d["crank_recovery_t"].notna()].sort_values("dtf", ascending=False)
        ax.plot(d["dtf"], d["crank_recovery_t"], color=BLUE, lw=1.4, marker="o", ms=3)
        ax.axhline(p95, color=RED, ls="--", lw=1.5, label=f"healthy-fleet p95 ({p95:.1f}s)")
        ax.invert_xaxis()
        ax.set_title(vin, fontweight="bold")
        ax.set_xlabel("days before failure")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("post-crank recovery time (s)")
    fig.suptitle("MVP signal: post-crank voltage recovery slows before failure (#3)",
                 fontweight="bold", y=1.02)
    return _save(fig, "G5_crank_recovery_trajectories")


def g6_changepoint_resting():
    cp = pd.read_csv(FOR / "changepoint_per_vin.csv")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    m = cp.melt(id_vars="vin_label",
                value_vars=["cp_resid_lead_days", "cp_resting_lead_days", "dose_knee_lead_days"],
                var_name="kind", value_name="lead")
    kinds = ["cp_resid_lead_days", "cp_resting_lead_days", "dose_knee_lead_days"]
    x = np.arange(len(cp)); w = 0.25
    for i, k in enumerate(kinds):
        vals = pd.to_numeric(cp[k], errors="coerce").fillna(0)
        axL.bar(x + (i - 1) * w, vals, width=w, label=k.replace("_lead_days", ""))
    axL.axhline(90, color=RED, ls="--", lw=1.5, label="actionable < 90d")
    axL.set_xticks(x); axL.set_xticklabels(cp["vin_label"], rotation=60, ha="right", fontsize=8)
    axL.set_ylabel("change-point lead (days)")
    axL.set_title("Change-point leads (#12/#5) — mostly fire too early (exploratory)", fontweight="bold")
    axL.legend(frameon=False, fontsize=8)
    rs = pd.to_numeric(cp["resting_slope"], errors="coerce")
    p05 = pd.to_numeric(cp["resting_slope_nf_p05"], errors="coerce").iloc[0]
    cols = [RED if b else GREY for b in cp["resting_slope_disc"]]
    axR.barh(cp["vin_label"], rs, color=cols)
    axR.axvline(p05, color=NAVY, ls="--", lw=1.5, label=f"NF p05 ({p05:.4f})")
    axR.set_xlabel("resting-voltage decay slope (V/day)")
    axR.set_title("Resting-voltage decay (#6) — red = discriminative", fontweight="bold")
    axR.legend(frameon=False, fontsize=8)
    return _save(fig, "G6_changepoint_resting")


def _write_report(written):
    md = ["# V11 Graphs — generation report\n",
          "_All graphs sourced from committed V11 caches; no pipeline re-runs._\n",
          "| graph | file | source |", "|---|---|---|",
          "| G1 recall | G1_recall_comparison.png | comparison.csv, nf_self_test, compound_alarm_lovo |",
          "| G2 feature generalization | G2_feature_generalization.png | failed_window_deviations, nf_self_test |",
          "| G3 lead-time dumbbell | G3_leadtime_dumbbell.png | comparison.csv |",
          "| G4 compound leads | G4_compound_alarm_leads.png | compound_alarm_lovo |",
          "| G5 crank-recovery (MVP) | G5_crank_recovery_trajectories.png | <VIN>_daily.csv, nf_baseline |",
          "| G6 changepoint/resting | G6_changepoint_resting.png | changepoint_per_vin |",
          "\n## Honesty notes\n",
          "- G5: crank_recovery_t within-truck z-scores are inflated by near-zero baseline "
          "variance + the 30s censor; the trustworthy guard is 0/15 NF false alarms.\n",
          "- G6: change-point fires far too early (220-599d) to be actionable — exploratory only.\n",
          "- No RUL/Weibull content: V11 has no RUL data (referenced to V10.6.2).\n"]
    (OUT / "Graphs_generation_report.md").write_text("\n".join(md), encoding="utf-8")
```

- [ ] **Step 2: Replace `main()`** with:
```python
def main():
    written = [g1_recall_comparison(), g2_feature_generalization(), g3_leadtime_dumbbell(),
               g4_compound_alarm_leads(), g5_crank_recovery_trajectories(), g6_changepoint_resting()]
    _write_report(written)
    print("[v11 graphs] wrote:", written)
    print("[v11 graphs] report:", OUT / "Graphs_generation_report.md")
```

- [ ] **Step 3: Run** — `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_graphs.py"`
Expected: prints 6 graph filenames; all PNG + _hd.png exist non-empty; `Graphs_generation_report.md` written. If any KeyError/empty-data, report the exact graph + error (do not fabricate data).

- [ ] **Step 4: Commit**
```
git add V11_ALT_heuristics/src/V11_ALT_heuristics_graphs.py V11_ALT_heuristics/visualizations/V11_graphs
git commit -m "feat(v11-alt): graphs G2/G4/G5/G6 + generation report (6-graph suite)"
```

---

## Task 3: Customer report (markdown + xlsx)

**Files:** Create `V11_ALT_heuristics/src/V11_ALT_heuristics_customer_report.py`.

- [ ] **Step 1: Create the module** — reads V11 caches, writes
`reports/V11_ALT_heuristics_customer_report.md` and
`reports/V11_ALT_heuristics_fleet_report.xlsx`.

```python
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
    md.append("- `crank_recovery_t` within-truck z-scores are inflated by near-zero baseline "
              "variance and the 30s recovery censor; trust the 0/15 NF false-alarm guard, not the "
              "z-magnitude.\n- VIN8's strict-gate hit rests on only 3 trusted days (fragile); the "
              "compound alarm catches it more robustly at 90d.\n- 4/10 failed trucks (VIN3/4/5/7) "
              "remain undetectable by any channel — consistent with abrupt/silent electrical "
              "failure with no precursor.\n- n=10 events: treat per-feature results as leads, not "
              "calibrated rates.\n- No per-truck daily RUL is produced or implied.\n")
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
```

- [ ] **Step 2: Run** — `py -3 "V11_ALT_heuristics\src\V11_ALT_heuristics_customer_report.py"`
Expected: writes the md + xlsx; prints both paths. Open the md and confirm sections 1–8 present and the per-truck + heuristic tables render. Confirm xlsx has 6 sheets (`py -3 -c "import openpyxl;wb=openpyxl.load_workbook(r'V11_ALT_heuristics/reports/V11_ALT_heuristics_fleet_report.xlsx');print(wb.sheetnames)"`).

- [ ] **Step 3: Commit**
```
git add V11_ALT_heuristics/src/V11_ALT_heuristics_customer_report.py V11_ALT_heuristics/reports
git commit -m "feat(v11-alt): customer report (md + xlsx), V11-native precursor-centric"
```

---

## Task 4: Technical presentation deck

**Files:** Create `V11_ALT_heuristics/presentation/build_technical_presentation.py`.

- [ ] **Step 1: Create the builder** (python-pptx; embeds graphs from `visualizations/V11_graphs/`). Use a helper for title+bullets slides and an image slide. ~13 slides per spec §5. Pull live numbers from the caches (do not hardcode recall) so the deck stays truthful.

```python
"""V11_ALT_heuristics — technical presentation (python-pptx), V11-native."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

_src = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = cfg.FORENSICS
GRAPHS = cfg.V11_ROOT / "visualizations" / "V11_graphs"
NAVY, GOLD, GREEN = RGBColor(0x0D, 0x1B, 0x2A), RGBColor(0xC5, 0x8B, 0x1F), RGBColor(0x27, 0xAE, 0x60)


def _nums():
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv")
    es = pd.read_csv(FOR / "earliest_signal_per_vin.csv")
    st = pd.read_csv(FOR / "nf_self_test.csv")
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    return {
        "n11": int((es["verdict"] == "discriminative_precursor").sum()),
        "n1062": int((comp["v1062_horizon"].astype(str) != "none").sum()),
        "nf_false": int((st["verdict"] == "FALSE_ALARM").sum()),
        "cf": int(cl[(cl.group == "FAILED") & (cl.fired == True)].shape[0]),   # noqa: E712
        "cnf": int(cl[(cl.group == "NF") & (cl.fired == True)].shape[0]),      # noqa: E712
    }


def _title_slide(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box = s.shapes.add_textbox(Inches(0.6), Inches(2.4), Inches(12), Inches(2))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = NAVY
    p2 = tf.add_paragraph(); p2.text = subtitle; p2.font.size = Pt(20); p2.font.color.rgb = GOLD
    return s


def _bullets_slide(prs, title, bullets):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(0.9))
    p = t.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = NAVY
    body = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.4)).text_frame
    body.word_wrap = True
    for i, b in enumerate(bullets):
        par = body.paragraphs[0] if i == 0 else body.add_paragraph()
        par.text = "• " + b; par.font.size = Pt(18); par.font.color.rgb = NAVY
    return s


def _image_slide(prs, title, img):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12), Inches(0.8))
    p = t.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = NAVY
    if pathlib.Path(img).exists():
        s.shapes.add_picture(str(img), Inches(1.2), Inches(1.2), height=Inches(5.6))
    return s


def main():
    n = _nums()
    prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
    _title_slide(prs, "Alternator Lead-Time Heuristics (V11)",
                 f"Recall {n['n11']}/10 vs V10.6.2 {n['n1062']}/10  |  {n['nf_false']}/15 false alarms")
    _bullets_slide(prs, "The problem V10.6.2 left open", [
        "Frozen classifier says WHICH truck (AUROC 0.927) but gives no timing.",
        "Per-truck RUL cannot beat a fleet-clock dummy (structural, n=10).",
        "Only genuine precursor was the GED=2 storm — fired for just 2/10 failures.",
        "Open question: can we get earlier, higher-recall precursors from the same 6 channels?"])
    _bullets_slide(prs, "Approach: 12 heuristics on 6 raw CAN channels", [
        "Load-normalised & regime-conditioned voltage (dVSI/dRPM, load residual, reg duty).",
        "Crank/recovery dynamics (post-crank recovery, crank effort).",
        "Excitation & instability (full GED states, idle hunting), sag typing, UV dose.",
        "Compound voting alarm (#11) + per-truck change-point (#12)."])
    _bullets_slide(prs, "Honest validation gate (unchanged from V10.6.2)", [
        "A deviation counts ONLY if within-truck z>=2 AND outside healthy-fleet p05-p95.",
        "MIN_EO_SAMPLES=200 trust filter; horizons 90/60/45/30/14/7 days.",
        "NF self-test: every healthy truck scored as-if-failing (false-alarm honesty).",
        "n=10 events — results are leads, not calibrated rates."])
    _image_slide(prs, "Result: lead-time recall", GRAPHS / "G1_recall_comparison.png")
    _image_slide(prs, "Per-truck head-to-head", GRAPHS / "G3_leadtime_dumbbell.png")
    _image_slide(prs, "MVP signal: post-crank recovery (#3)", GRAPHS / "G5_crank_recovery_trajectories.png")
    _image_slide(prs, "Which heuristics generalised", GRAPHS / "G2_feature_generalization.png")
    _image_slide(prs, "Compound early-watch alarm (#11)", GRAPHS / "G4_compound_alarm_leads.png")
    _image_slide(prs, "Change-point / resting voltage (exploratory)", GRAPHS / "G6_changepoint_resting.png")
    _bullets_slide(prs, "Limitations", [
        f"Net gain is modest: +1 detection (VIN9), +1 earlier lead (VIN1).",
        "crank_recovery_t z-magnitudes inflated; trust the 0/15 NF guard not the z.",
        "VIN8 strict-gate hit rests on 3 trusted days (compound catches it at 90d).",
        "4/10 failures remain undetectable (abrupt/silent electrical failure).",
        "No per-truck daily RUL produced or implied."])
    _bullets_slide(prs, "Deployment recommendation", [
        "Add crank_recovery_t to the emergency channel alongside the GED=2 storm.",
        "Surface the compound 2-of-5 vote as an 'early-watch' tier (0 false alarms).",
        "Use sag-typing (high-load vs idle) + reg-duty for repair-direction guidance.",
        "WHICH (classifier) + WHEN-fleet (Weibull) unchanged from V10.6.2."])
    _bullets_slide(prs, "Appendix — data sources", [
        "All numbers from V11 cache/forensics + results/comparison.csv.",
        "No RUL/Weibull/backtest data (V11 is the precursor fork; see V10.6.2 for RUL).",
        "Graphs: visualizations/V11_graphs/.",
        "Pipeline reproducible via V11_ALT_heuristics_orchestrator.py."])
    out = cfg.V11_ROOT / "presentation" / "Alternator_LeadTime_Heuristics_V11.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"[v11 technical deck] wrote {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `py -3 "V11_ALT_heuristics\presentation\build_technical_presentation.py"`
Expected: writes `Alternator_LeadTime_Heuristics_V11.pptx`; prints a slide count (~13). If the slide-count print line errors, simplify it to `print("wrote", out)` — the deck save is what matters. Confirm file exists non-empty.

- [ ] **Step 3: Commit**
```
git add V11_ALT_heuristics/presentation/build_technical_presentation.py V11_ALT_heuristics/presentation/Alternator_LeadTime_Heuristics_V11.pptx
git commit -m "feat(v11-alt): technical presentation deck (python-pptx, embeds G1-G6)"
```

---

## Task 5: Business deck + DATA_SOURCES.md + AUDIT_REPORT.md

**Files:** Create `V11_ALT_heuristics/presentation/build_business_presentation.py`, `V11_ALT_heuristics/presentation/DATA_SOURCES.md`, `V11_ALT_heuristics/presentation/AUDIT_REPORT.md`.

- [ ] **Step 1: Create the business builder** (reuse the helper pattern; 5 slides). It may import the helpers by copying the three helper functions (`_title_slide`, `_bullets_slide`, `_image_slide`, `_nums`) — keep it self-contained.

```python
"""V11_ALT_heuristics — business summary deck (5 slides), V11-native."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

_src = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FOR = cfg.FORENSICS
GRAPHS = cfg.V11_ROOT / "visualizations" / "V11_graphs"
NAVY, GOLD = RGBColor(0x0D, 0x1B, 0x2A), RGBColor(0xC5, 0x8B, 0x1F)


def _nums():
    comp = pd.read_csv(cfg.RESULTS_DIR / "V11_ALT_heuristics_comparison.csv")
    es = pd.read_csv(FOR / "earliest_signal_per_vin.csv")
    st = pd.read_csv(FOR / "nf_self_test.csv")
    cl = pd.read_csv(FOR / "compound_alarm_lovo.csv")
    return {"n11": int((es["verdict"] == "discriminative_precursor").sum()),
            "n1062": int((comp["v1062_horizon"].astype(str) != "none").sum()),
            "nf_false": int((st["verdict"] == "FALSE_ALARM").sum()),
            "cf": int(cl[(cl.group == "FAILED") & (cl.fired == True)].shape[0])}  # noqa: E712


def _bullets(prs, title, bullets):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(0.9))
    p = t.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = NAVY
    body = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.6), Inches(5.2)).text_frame
    body.word_wrap = True
    for i, b in enumerate(bullets):
        par = body.paragraphs[0] if i == 0 else body.add_paragraph()
        par.text = "• " + b; par.font.size = Pt(20); par.font.color.rgb = NAVY
    return s


def main():
    n = _nums()
    prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box = s.shapes.add_textbox(Inches(0.6), Inches(2.6), Inches(12), Inches(2))
    p = box.text_frame.paragraphs[0]; p.text = "Alternator Lead-Time Heuristics — Business Summary (V11)"
    p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = NAVY
    p2 = box.text_frame.add_paragraph()
    p2.text = f"Earlier warnings for more trucks, zero false alarms ({n['n11']}/10 vs {n['n1062']}/10, {n['nf_false']}/15 FP)"
    p2.font.size = Pt(18); p2.font.color.rgb = GOLD
    _bullets(prs, "How we did it", [
        "Engineered 12 new lead-time signals from the existing 6 CAN channels — no new sensors.",
        "Validated each against the same honest gate (change vs the truck's own healthy baseline AND vs the healthy fleet).",
        "Added a compound 'early-watch' vote and per-truck change-point detection."])
    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    t = s3.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12), Inches(0.8))
    pp = t.text_frame.paragraphs[0]; pp.text = "Results & impact"; pp.font.size = Pt(30); pp.font.bold = True; pp.font.color.rgb = NAVY
    if (GRAPHS / "G1_recall_comparison.png").exists():
        s3.shapes.add_picture(str(GRAPHS / "G1_recall_comparison.png"), Inches(1.5), Inches(1.2), height=Inches(5.6))
    _bullets(prs, "Limitations & data gaps", [
        "Gain is real but modest: one more truck detected, one earlier warning.",
        "4 of 10 failures have no electrical precursor (abrupt/silent) — undetectable from this data.",
        "No change to per-truck remaining-life dates (structural limit at n=10).",
        "Results are leads to act on, not calibrated probabilities."])
    _bullets(prs, "Next steps", [
        "Deploy post-crank recovery + compound early-watch in the service emergency channel.",
        "Use load-split sag typing for repair-direction guidance.",
        "Re-validate on the larger starter-motor fleet (n=34) and as more failures accrue."])
    out = cfg.V11_ROOT / "presentation" / "Alternator_Business_Summary_V11.pptx"
    prs.save(str(out)); print(f"[v11 business deck] wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `py -3 "V11_ALT_heuristics\presentation\build_business_presentation.py"` → writes the business pptx.

- [ ] **Step 3: Write `DATA_SOURCES.md` and `AUDIT_REPORT.md`** in `V11_ALT_heuristics/presentation/` — short, honest:
  - DATA_SOURCES.md: pipeline (raw CAN → forensic gate → compound/changepoint → comparison), the exact V11 cache files each deck/graph reads, and an explicit "RUL/Weibull out of scope (V11 is the precursor fork)" note.
  - AUDIT_REPORT.md: every numeric claim in the decks (recall 6/10, 5/10, 0/15, compound 4/10, MVP crank_recovery_t) traced to its cache file; list embedded graphs G1–G6.

- [ ] **Step 4: Commit**
```
git add V11_ALT_heuristics/presentation
git commit -m "feat(v11-alt): business deck + DATA_SOURCES + AUDIT_REPORT"
```

---

## Task 6: Final verification

- [ ] **Step 1: Inventory** — confirm all artifacts exist non-empty:
`py -3 -c "import pathlib,glob; [print(p) for p in sorted(glob.glob('V11_ALT_heuristics/visualizations/V11_graphs/*'))+sorted(glob.glob('V11_ALT_heuristics/reports/V11_ALT_heuristics_*'))+sorted(glob.glob('V11_ALT_heuristics/presentation/*.pptx'))]"`
Expected: 6 PNG + 6 _hd.png + Graphs_generation_report.md; customer_report.md + fleet_report.xlsx; 2 pptx.

- [ ] **Step 2: Honesty cross-check** — open the customer report md and confirm: recall stated as 6/10 vs 5/10, 0/15 false alarms, the 4/10-undetectable and no-RUL caveats present, no fabricated RUL/Weibull numbers anywhere.

- [ ] **Step 3: Final commit** (if any stragglers)
```
git add V11_ALT_heuristics
git commit -m "chore(v11-alt): finalize V11 deliverables (graphs, report, decks)" || echo "nothing to commit"
```

---

## Self-Review

- Spec coverage: graphs G1–G6 (spec §3) → Tasks 1–2; report md+xlsx (§4) → Task 3; technical deck (§5) → Task 4; business deck + DATA_SOURCES + AUDIT (§5) → Task 5; verification (§7) → Task 6. ✓
- Placeholder scan: all code complete; run/expected lines concrete. ✓
- Name consistency: `cfg.V11_ROOT`, `cfg.FORENSICS`, `cfg.RESULTS_DIR`, `cfg.REPORTS_DIR`, `cfg.NEW_FEATS`, `cfg.MIN_EO_SAMPLES` are real config attributes (verified in V11 config). Graph filenames G1..G6 referenced consistently by the deck builders. ✓
- Known risk: the technical deck's slide-count print uses a fragile internal (`prs.slides.__iter__...`); Task 4 Step 2 already instructs to simplify to `print("wrote", out)` if it errors. Not a blocker.
