"""
V11.1_ALT — Honest Verification Gates
======================================
Five gates that must ALL pass before the pipeline can be declared clean.

Gate catalogue
--------------
G-LEAK  : Recompute (x1, x2) for every (vin_label, horizon) row of the
          CHOSEN variant (and the strictest covariate variant, M2) using
          B.rewind_covariates() and compare to the logged values with
          abs tol 1e-9.  Any mismatch signals calendar leakage.

G-BETA  : If chosen_variant != "M0": require g_beta_ships(chosen, M0,
          mae_dummy).  If chosen == "M0": no improvement — require that
          NEITHER M1 nor M2 satisfies g_beta_ships; record
          NO_IMPROVEMENT_HONEST.

G-W6    : scan_forbidden() over every V11.1_ALT/src/V11_1_ALT_*.py file
          EXCEPT the config (which lists the patterns literally).  Any
          forbidden pattern in production source fails.

G-EMERG : From emergency_per_vin.csv — failed ged_fired sum >= 2 AND
          NF ged_fired sum == 0 AND NF early_watch_current sum == 0 AND
          failed early_watch_current sum >= 3.
          Ever-fired exceed/compound NF rates are recorded as report-only
          evidence and are NOT gated.

G-COVER : Chosen variant pi_coverage >= 0.80.

All failures are collected; exit(1) if any fail, else PASS.
Results written to cfg.RESULTS_DIR_V11_1 / "V11.1_ALT_verification.json".
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config")
C = _load("V11_1_ALT_covariates")
B = _load("V11_1_ALT_backtest")


# ---------------------------------------------------------------------------
# Pure helper functions (imported by tests)
# ---------------------------------------------------------------------------

def g_beta_ships(variant: dict, m0: dict, dummy_mae: float) -> bool:
    """A covariate variant may ship only if it (a) beats BOTH the dummy and M0
    on MAE with signed-rank p < alpha, or (b) shrinks mean PI width >= 15% vs M0
    at >= 80% coverage."""
    a = (variant["mae_model"] < dummy_mae and variant["mae_model"] < m0["mae_model"]
         and variant.get("wilcoxon_p_vs_dummy", 1.0) < cfg.SIGNED_RANK_ALPHA
         and variant.get("wilcoxon_p_vs_m0", 1.0) < cfg.SIGNED_RANK_ALPHA)
    b = (variant["mean_pi_width"] <= (1.0 - cfg.PI_WIDTH_SHRINK_MIN) * m0["mean_pi_width"]
        and variant["pi_coverage"] >= 0.80)
    return bool(a or b)


def scan_forbidden(text: str) -> list:
    """Return list of forbidden patterns found in text."""
    return [p for p in cfg.FORBIDDEN_FIT_PATTERNS if p in text]


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------

def _gate_leak(residuals: pd.DataFrame, chosen: str, p95: float, nfb: pd.DataFrame) -> tuple:
    """G-LEAK: recompute covariates and compare to logged values."""
    failures = []
    n_checked = 0

    # Check the chosen variant rows and M2 rows (strictest covariate user).
    # Use a set union so we don't double-check when chosen == "M2".
    variants_to_check = {chosen, "M2"}

    for variant in variants_to_check:
        sub = residuals[residuals["variant"] == variant]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            vin = row["vin_label"]
            ttf = float(row["ttf"])
            horizon = float(row["horizon"])
            logged_x1 = float(row["x1"])
            logged_x2 = float(row["x2"])

            x1_recomp, x2_recomp = B.rewind_covariates(
                vin, ttf, horizon, C.covariate_vector, p95, nfb
            )
            n_checked += 1

            # The backtest stores x1 rounded to 4dp (round(x1_v, 4)) and x2 as int.
            # Match at that precision before applying the 1e-9 tolerance.
            x1_ok = abs(round(float(x1_recomp), 4) - logged_x1) <= 1e-9
            x2_ok = abs(float(x2_recomp) - logged_x2) <= 1e-9

            if not x1_ok:
                failures.append(
                    f"G-LEAK MISMATCH x1: {vin} H={horizon} "
                    f"logged={logged_x1} recomp={x1_recomp:.10f}"
                )
            if not x2_ok:
                failures.append(
                    f"G-LEAK MISMATCH x2: {vin} H={horizon} "
                    f"logged={logged_x2} recomp={x2_recomp}"
                )

    print(f"  [G-LEAK] Checked {n_checked} rows (variants: {sorted(variants_to_check)})")
    return failures, n_checked


def _gate_beta(variants_dict: dict, chosen: str) -> tuple:
    """G-BETA: variant selection honesty check."""
    failures = []
    verdict = ""

    m0_stats = variants_dict.get("M0", {})
    dummy_mae = m0_stats.get("mae_dummy", None)

    if dummy_mae is None:
        failures.append("G-BETA: mae_dummy not found in M0 stats")
        return failures, "ERROR"

    if chosen != "M0":
        chosen_stats = variants_dict.get(chosen, {})
        if not g_beta_ships(chosen_stats, m0_stats, dummy_mae):
            failures.append(
                f"G-BETA: chosen variant {chosen} was selected but fails "
                f"g_beta_ships gate (neither criterion satisfied)"
            )
            verdict = "SHIPS_FAIL"
        else:
            verdict = "SHIPS_PASS"
    else:
        # chosen == "M0": confirm that no covariate variant earns the gate
        better = []
        for vt in ["M1", "M2"]:
            vt_stats = variants_dict.get(vt)
            if vt_stats is None:
                continue
            if g_beta_ships(vt_stats, m0_stats, dummy_mae):
                better.append(vt)
        if better:
            failures.append(
                f"G-BETA: chosen_variant=M0 but {better} satisfy g_beta_ships "
                f"— selection contradicts the gate"
            )
            verdict = "CONTRADICTED"
        else:
            verdict = "NO_IMPROVEMENT_HONEST"

    return failures, verdict


def _gate_w6() -> list:
    """G-W6: scan src files for forbidden fit patterns."""
    failures = []
    src_dir = _src
    config_name = "V11_1_ALT_config.py"
    for py_file in sorted(src_dir.glob("V11_1_ALT_*.py")):
        if py_file.name == config_name:
            continue
        text = py_file.read_text(encoding="utf-8")
        hits = scan_forbidden(text)
        if hits:
            failures.append(
                f"G-W6: forbidden pattern(s) {hits} found in {py_file.name}"
            )
    return failures


def _gate_emerg(emerg: pd.DataFrame) -> tuple:
    """G-EMERG: emergency channel integrity check."""
    failures = []

    # Normalise boolean columns that may have been read as strings
    def _to_bool(col):
        return col.map(lambda v: v if isinstance(v, bool) else str(v).strip() == "True")

    emerg = emerg.copy()
    for col in ["ged_fired", "exceed_fired", "compound_fired"]:
        if col in emerg.columns:
            emerg[col] = _to_bool(emerg[col])

    failed = emerg[emerg["failed_flag"] == 1]
    nf = emerg[emerg["failed_flag"] == 0]

    failed_ged = int(failed["ged_fired"].sum())
    nf_ged = int(nf["ged_fired"].sum())
    failed_ew = int(failed["early_watch_current"].sum())
    nf_ew = int(nf["early_watch_current"].sum())

    # Report-only evidence (not gated)
    nf_exceed_fired = int(_to_bool(nf["exceed_fired"]).sum()) if "exceed_fired" in nf.columns else None
    nf_compound_fired = int(_to_bool(nf["compound_fired"]).sum()) if "compound_fired" in nf.columns else None

    if failed_ged < 2:
        failures.append(
            f"G-EMERG: failed ged_fired sum={failed_ged} < 2 (expected >= 2)"
        )
    if nf_ged != 0:
        failures.append(
            f"G-EMERG: NF ged_fired sum={nf_ged} != 0 (non-deployable channel)"
        )
    if nf_ew != 0:
        failures.append(
            f"G-EMERG: NF early_watch_current sum={nf_ew} != 0 (false positives)"
        )
    if failed_ew < 3:
        failures.append(
            f"G-EMERG: failed early_watch_current sum={failed_ew} < 3 (too few detections)"
        )

    evidence = {
        "failed_ged_fired": failed_ged,
        "nf_ged_fired": nf_ged,
        "failed_early_watch_current": failed_ew,
        "nf_early_watch_current": nf_ew,
        "nf_exceed_fired_report_only": nf_exceed_fired,
        "nf_compound_fired_report_only": nf_compound_fired,
    }
    return failures, evidence


def _gate_cover(chosen_stats: dict) -> list:
    """G-COVER: chosen variant predictive interval coverage >= 0.80."""
    failures = []
    cov = chosen_stats.get("pi_coverage", 0.0)
    if cov < 0.80:
        failures.append(
            f"G-COVER: pi_coverage={cov:.3f} < 0.80 for chosen variant"
        )
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.RESULTS_DIR_V11_1.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("[v11.1 verify] Running five honest gates")
    print("=" * 70)

    # ---- Load shared inputs ------------------------------------------------
    p95 = float((cfg.COV_CACHE / "crank_p95.txt").read_text(encoding="utf-8").strip())
    nfb = pd.read_csv(cfg.V11_FORENSICS / "nf_baseline.csv").set_index("feature")

    residuals = pd.read_csv(cfg.BACKTEST_CACHE / "per_fold_residuals.csv")

    with open(cfg.BACKTEST_CACHE / "backtest_results.json") as f:
        bt = json.load(f)
    variants_dict = bt["variants"]
    chosen = bt["chosen_variant"]
    selection_rule = bt["selection_rule"]

    emerg = pd.read_csv(cfg.EMERG_CACHE / "emergency_per_vin.csv")

    # ---- Run gates ---------------------------------------------------------
    all_failures = []
    gate_results = {}

    # G-LEAK
    print("\n[G-LEAK] Recomputing covariates vs logged values ...")
    leak_failures, n_checked = _gate_leak(residuals, chosen, p95, nfb)
    gate_results["G-LEAK"] = {
        "status": "PASS" if not leak_failures else "FAIL",
        "n_checked": n_checked,
        "failures": leak_failures,
    }
    if leak_failures:
        for f in leak_failures:
            print(f"  FAIL: {f}")
        all_failures.extend(leak_failures)
    else:
        print(f"  PASS — {n_checked} rows verified, all within abs tol 1e-9")

    # G-BETA
    print("\n[G-BETA] Checking variant selection honesty ...")
    beta_failures, beta_verdict = _gate_beta(variants_dict, chosen)
    gate_results["G-BETA"] = {
        "status": "PASS" if not beta_failures else "FAIL",
        "chosen_variant": chosen,
        "selection_rule": selection_rule,
        "verdict": beta_verdict,
    }
    if beta_failures:
        for f in beta_failures:
            print(f"  FAIL: {f}")
        all_failures.extend(beta_failures)
    else:
        print(f"  PASS — verdict: {beta_verdict}")

    # G-W6
    print("\n[G-W6] Scanning source files for forbidden fit patterns ...")
    w6_failures = _gate_w6()
    gate_results["G-W6"] = {
        "status": "PASS" if not w6_failures else "FAIL",
        "forbidden_patterns": cfg.FORBIDDEN_FIT_PATTERNS,
        "failures": w6_failures,
    }
    if w6_failures:
        for f in w6_failures:
            print(f"  FAIL: {f}")
        all_failures.extend(w6_failures)
    else:
        print(f"  PASS — no forbidden patterns found in V11_1_ALT_*.py (config excluded)")

    # G-EMERG
    print("\n[G-EMERG] Checking emergency channel integrity ...")
    emerg_failures, emerg_evidence = _gate_emerg(emerg)
    gate_results["G-EMERG"] = {
        "status": "PASS" if not emerg_failures else "FAIL",
        "evidence": emerg_evidence,
        "failures": emerg_failures,
    }
    if emerg_failures:
        for f in emerg_failures:
            print(f"  FAIL: {f}")
        all_failures.extend(emerg_failures)
    else:
        ev = emerg_evidence
        print(
            f"  PASS — failed ged_fired={ev['failed_ged_fired']} "
            f"NF ged_fired={ev['nf_ged_fired']} "
            f"failed early_watch={ev['failed_early_watch_current']} "
            f"NF early_watch={ev['nf_early_watch_current']}"
        )
        print(
            f"  [report-only] NF exceed_fired={ev['nf_exceed_fired_report_only']} "
            f"NF compound_fired={ev['nf_compound_fired_report_only']} (non-deployable channels)"
        )

    # G-COVER
    print("\n[G-COVER] Checking chosen variant PI coverage ...")
    chosen_stats = variants_dict.get(chosen, {})
    cover_failures = _gate_cover(chosen_stats)
    cov_val = chosen_stats.get("pi_coverage", 0.0)
    gate_results["G-COVER"] = {
        "status": "PASS" if not cover_failures else "FAIL",
        "chosen_variant": chosen,
        "pi_coverage": cov_val,
    }
    if cover_failures:
        for f in cover_failures:
            print(f"  FAIL: {f}")
        all_failures.extend(cover_failures)
    else:
        print(f"  PASS — {chosen} pi_coverage={cov_val:.3f} >= 0.80")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    overall = "PASS" if not all_failures else "FAIL"
    for gname, gres in gate_results.items():
        status = gres["status"]
        marker = "  OK" if status == "PASS" else "FAIL"
        print(f"  {marker}  {gname}")
    print("=" * 70)
    print(f"[v11.1 verify] {overall}")

    # ---- Write JSON --------------------------------------------------------
    out = {
        "overall": overall,
        "chosen_variant": chosen,
        "selection_rule": selection_rule,
        "gates": gate_results,
        "total_failures": len(all_failures),
    }
    out_path = cfg.RESULTS_DIR_V11_1 / "V11.1_ALT_verification.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Written: {out_path}")

    if all_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
