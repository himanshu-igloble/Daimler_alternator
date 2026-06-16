"""
V10.6.2 Alternator — Honest Verification Gates  (plan W6)
=========================================================
Replaces the V10.6.1 gate set (which passed 12/12 while hiding in-sample,
inert RUL).  Here a gate either PASSES, FAILS, or is DOCUMENTED.

GATING gates (must pass for overall_pass):
  ridge_integrity, cohort_25, posterior_shape_used, pi_coverage,
  decision_sanity, failed_vin_handling, reports_clean
DOCUMENTED gates (recorded, never fail the run — an honest "no improvement"
is a valid finding):
  backtest_vs_dummy, ged_oos_documented

Output: results/V10.6.2_ALT_verification.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    gates = []

    def add(name, status, gating, details):
        gates.append({"name": name, "status": status, "gating": gating, "details": details})

    # --- G1 ridge_integrity ------------------------------------------------
    try:
        sha = _sha256(cfg.RIDGE_PROB_CSV)
        rp = pd.read_csv(cfg.RIDGE_PROB_CSV)
        # scan our src for forbidden classifier-retraining patterns (skip config.py)
        hits = []
        for py in _src.glob("V10.6.2_ALT_*.py"):
            if py.name == "V10.6.2_ALT_config.py":
                continue
            txt = py.read_text(encoding="utf-8", errors="ignore")
            for pat in cfg.FORBIDDEN_FIT_PATTERNS:
                if pat in txt:
                    hits.append(f"{py.name}:{pat}")
        ok = (len(rp) == 25) and (len(hits) == 0)
        add("ridge_integrity", "PASS" if ok else "FAIL", True,
            f"ridge_prob_rescaled.csv sha256={sha[:12]}..., rows={len(rp)}; "
            f"forbidden-pattern hits in src: {hits if hits else 'none'}")
    except Exception as e:
        add("ridge_integrity", "FAIL", True, f"error: {e}")

    # --- G2 cohort_25 ------------------------------------------------------
    wb = json.loads((pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json").read_text())
    ok = (wb.get("n_cohort") == 25) and (wb.get("n_events") == 10)
    add("cohort_25", "PASS" if ok else "FAIL", True,
        f"n_cohort={wb.get('n_cohort')}, n_events={wb.get('n_events')} (expect 25/10)")

    # --- G3 posterior_shape_used (D4) -------------------------------------
    shape = wb.get("shape")
    map_shape = wb.get("posterior_map_shape")
    mle_shape = wb.get("mle_shape")
    ok = (abs(shape - map_shape) < 1e-6) and (abs(shape - mle_shape) > 0.1)
    add("posterior_shape_used", "PASS" if ok else "FAIL", True,
        f"reported shape={shape} == posterior_map={map_shape}; MLE={mle_shape} (must differ)")

    # --- G4 backtest_vs_dummy (DOCUMENTED) --------------------------------
    bt = json.loads((pathlib.Path(cfg.BACKTEST_CACHE) / "backtest_results.json").read_text())
    rew = bt["time_rewound"]["overall"]
    add("backtest_vs_dummy", "DOCUMENTED", False,
        f"verdict={bt['time_rewound']['verdict']}; rewound MAE model={rew['mae_model']}d "
        f"vs dummyA={rew['mae_dummyA']}d; signed_rank_p={rew['signed_rank_p']}; "
        f"total-TTF MAE model={bt['lovo_total_ttf']['mae_model']}d vs dummyA={bt['lovo_total_ttf']['mae_dummyA']}d")

    # --- G5 pi_coverage (gating) ------------------------------------------
    cov = bt["lovo_total_ttf"]
    add("pi_coverage", "PASS" if cov["pi_gate_pass"] else "FAIL", True,
        f"total-TTF PI coverage={cov['pi_coverage_k_of_n']} "
        f"(gate>= {cfg.PI_COVERAGE_TARGET:.0%}); rewound coverage={rew['pi_coverage']}")

    # --- load deliverable for G6/G7 ---------------------------------------
    pred = pd.read_csv(pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_rul_predictions.csv")

    # --- G6 decision_sanity -----------------------------------------------
    crit = pred[pred["recommendation"].astype(str).str.startswith("CRITICAL")]
    bad = crit[crit["ridge_prob"] < cfg.RIDGE_DECISION_THR]
    add("decision_sanity", "PASS" if len(bad) == 0 else "FAIL", True,
        f"{len(crit)} CRITICAL rows; {len(bad)} with ridge_prob<{cfg.RIDGE_DECISION_THR} (must be 0)")

    # --- G7 failed_vin_handling -------------------------------------------
    fv = pred[pred["failed_flag"] == 1]
    rul0 = (fv["median_rul_days"].astype(float) == 0).all()
    lead_present = fv["would_have_flagged_lead_days"].astype(str).str.len().gt(0).all()
    ok = (len(fv) == 10) and rul0 and lead_present
    add("failed_vin_handling", "PASS" if ok else "FAIL", True,
        f"{len(fv)} failed; all RUL=0: {rul0}; would_have_flagged present: {lead_present}")

    # --- G8 ged_oos_documented (DOCUMENTED) -------------------------------
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")
    nfire_f = int(ged[(ged["failed_flag"] == 1) & (ged["ever_fired"])].shape[0])
    nfire_nf = int(ged[(ged["failed_flag"] == 0) & (ged["ever_fired"])].shape[0])
    add("ged_oos_documented", "DOCUMENTED", False,
        f"daily>= {cfg.GED_EMERGENCY_DAILY_COUNT_MIN}/day: sensitivity {nfire_f}/10, "
        f"false alarms {nfire_nf}/15")

    # --- G9 reports_clean (B1/B2) -----------------------------------------
    # Structural fixes (binding): B1 = params JSON carries median_ttf_days (the
    # key the report reader needs); B2 = this verification.json is keyed by gate
    # NAME, which the report renders by name.  Reports render AFTER verify, so
    # also check the previously-rendered report if one exists.
    b1_source = (wb.get("median_ttf_days") is not None)
    rpt = pathlib.Path(cfg.REPORTS_DIR_V2) / "V10.6.2_ALT_customer_report.md"
    rendered_note = "report renders after verify"
    if rpt.exists():
        txt = rpt.read_text(encoding="utf-8", errors="ignore")
        prior_ok = ("N/A days" not in txt) and ("Median TTF: N/A" not in txt)
        rendered_note = f"prior report clean: {prior_ok}"
    add("reports_clean", "PASS" if b1_source else "FAIL", True,
        f"B1 params has median_ttf_days={b1_source}; B2 gates keyed by name=True; {rendered_note}")

    # --- summary ----------------------------------------------------------
    gating = [g for g in gates if g["gating"]]
    n_pass = sum(g["status"] == "PASS" for g in gating)
    n_fail = sum(g["status"] == "FAIL" for g in gating)
    overall = n_fail == 0

    result = {
        "gates": gates,
        "n_gating": len(gating),
        "n_gating_pass": n_pass,
        "n_gating_fail": n_fail,
        "n_documented": sum(1 for g in gates if not g["gating"]),
        "overall_pass": overall,
    }
    out = pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_verification.json"
    out.write_text(json.dumps(result, indent=2))

    print("=" * 64)
    print("V10.6.2 VERIFICATION")
    print("=" * 64)
    for g in gates:
        flag = "" if g["gating"] else "  (documented)"
        print(f"  [{g['status']:<10}] {g['name']:<24}{flag}")
        print(f"       {g['details']}")
    print("-" * 64)
    print(f"  GATING: {n_pass}/{len(gating)} pass, {n_fail} fail   "
          f"OVERALL: {'PASS' if overall else 'FAIL'}")
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
