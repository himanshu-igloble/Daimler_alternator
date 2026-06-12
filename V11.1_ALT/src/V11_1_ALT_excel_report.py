"""
V11.1_ALT — Excel Workbook  (Tasks 10+11)
==========================================
Sheets:
  Summary          — key metrics, chosen variant, gate overview
  Risk_Ranking     — NF trucks sorted by ridge_prob (frozen classifier)
  RUL_Band         — all 25 trucks RUL band (NF actionable, F historical)
  Covariates       — covariates_fit with confound note row appended
  Variant_Comparison — 3-variant MAE / coverage / PI-width
  Emergency        — per-vin emergency channel table
  Verification     — five gates with status and details

Output: reports/V11.1_ALT_fleet_report.xlsx
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent
_root = _src.parent
_v1062_root = _root.parent / "V10.6.2_ALT"


def _load(mod_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config", _src / "V11_1_ALT_config.py")


def main() -> None:
    # ── load caches ───────────────────────────────────────────────────────────
    wb_params = json.loads((cfg.WEIBULL_CACHE / "fleet_weibull_params.json").read_text())
    fw = json.loads((_v1062_root / "cache" / "rul" / "fleet_window.json").read_text())
    bt = json.loads((cfg.BACKTEST_CACHE / "backtest_results.json").read_text())
    ver = json.loads((cfg.RESULTS_DIR_V11_1 / "V11.1_ALT_verification.json").read_text())
    emg = pd.read_csv(cfg.EMERG_CACHE / "emergency_per_vin.csv")
    final = pd.read_csv(cfg.RUL_CACHE / "final_rul_per_vin.csv")
    cov = pd.read_csv(cfg.COV_CACHE / "covariates_fit.csv")

    m0 = bt["variants"]["M0"]
    m1 = bt["variants"]["M1"]
    m2 = bt["variants"]["M2"]

    ged_failed  = int(emg[(emg.failed_flag == 1) & (emg.ged_fired == True)].shape[0])
    ged_nf      = int(emg[(emg.failed_flag == 0) & (emg.ged_fired == True)].shape[0])
    ew_failed   = int(emg[(emg.failed_flag == 1) & (emg.early_watch_current == 1)].shape[0])
    ew_nf       = int(emg[(emg.failed_flag == 0) & (emg.early_watch_current == 1)].shape[0])

    # ── Sheet: Summary ────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "metric": [
            "version",
            "classifier_AUROC_frozen",
            "chosen_variant",
            "covariate_verdict",
            "M0_mae_model_days",
            "M1_mae_model_days",
            "M2_mae_model_days",
            "dummy_mae_days",
            "M0_pi_coverage",
            "fleet_empirical_median_ttf_days",
            "fleet_window_p25_p75_days",
            "fleet_empirical_median_km_est",
            "fleet_empirical_median_ehrs_est",
            "weibull_median_context_days",
            "weibull_shape_posterior",
            "weibull_scale",
            "ged_sensitivity_failed",
            "ged_false_alarms_nf",
            "early_watch_current_failed",
            "early_watch_current_nf",
            "overall_gate_status",
        ],
        "value": [
            cfg.VERSION,
            "0.927",
            bt["chosen_variant"],
            ver["gates"]["G-BETA"]["verdict"],
            m0["mae_model"],
            m1["mae_model"],
            m2["mae_model"],
            m0["mae_dummy"],
            m0["pi_coverage"],
            fw["median_ttf_days"],
            f"{fw['p25_ttf_days']:.0f}–{fw['p75_ttf_days']:.0f}",
            fw["median_ttf_km_est"],
            fw["median_ttf_ehrs_est"],
            wb_params["median_ttf_days"],
            wb_params["shape"],
            wb_params["scale"],
            f"{ged_failed}/10",
            f"{ged_nf}/15",
            f"{ew_failed}/10",
            f"{ew_nf}/15",
            ver["overall"],
        ],
    })

    # ── Sheet: Risk_Ranking ───────────────────────────────────────────────────
    nf = final[final["failed_flag"] == 0].sort_values("ridge_prob", ascending=False)
    risk = nf[["vin_label", "ridge_prob", "risk_band", "above_thr",
               "emergency_state", "recommendation"]].copy()

    # ── Sheet: RUL_Band ───────────────────────────────────────────────────────
    rul_band = final[["vin_label", "failed_flag", "lifecycle_stage",
                      "current_age_days", "median_rul_days",
                      "rul_p10_days", "rul_p90_days", "pi_width_days",
                      "risk_band", "time_dim", "emergency_state",
                      "recommendation"]].copy()

    # ── Sheet: Covariates ─────────────────────────────────────────────────────
    # Append a confound-note sentinel row
    cov_with_note = pd.concat([
        cov,
        pd.DataFrame([{
            "vin_label": "NOTE",
            "failed_flag": "",
            "t_end": "",
            "x1": "x1 = log(lifetime exceedance count). NF trucks accumulate higher x1 simply because they live longer (exposure confound). See §2.",
            "x2": "x2 = trailing compound-vote indicator. Both covariates fail G-BETA.",
        }])
    ], ignore_index=True)

    # ── Sheet: Variant_Comparison ─────────────────────────────────────────────
    variant_rows = []
    for vname, vm, desc in [
        ("M0", m0, "Baseline — no covariates; M0 ≡ V10.6.2 fleet curve"),
        ("M1", m1, "x1 = log lifetime GED+compound exceedances (exposure-confounded)"),
        ("M2", m2, "x1 + x2 trailing compound-vote"),
    ]:
        h270 = vm["per_horizon_mae"].get("270", "")
        h180 = vm["per_horizon_mae"].get("180", "")
        h90  = vm["per_horizon_mae"].get("90", "")
        variant_rows.append({
            "variant": vname,
            "description": desc,
            "mae_model_days": vm["mae_model"],
            "mae_dummy_days": vm["mae_dummy"],
            "pi_coverage": vm["pi_coverage"],
            "mean_pi_width_days": vm["mean_pi_width"],
            "mae_T-270d": h270,
            "mae_T-180d": h180,
            "mae_T-90d": h90,
            "wilcoxon_p_vs_dummy": vm["wilcoxon_p_vs_dummy"],
            "wilcoxon_p_vs_m0": vm.get("wilcoxon_p_vs_m0", "N/A"),
            "selected": vname == bt["chosen_variant"],
        })
    variants_df = pd.DataFrame(variant_rows)

    # ── Sheet: Emergency ──────────────────────────────────────────────────────
    emg_out = emg.copy()

    # ── Sheet: Verification ───────────────────────────────────────────────────
    gate_rows = []
    for gname, gdata in ver["gates"].items():
        gate_rows.append({
            "gate": gname,
            "status": gdata["status"],
            "details": json.dumps(
                {k: v for k, v in gdata.items() if k != "status"}, default=str
            ),
        })
    ver_df = pd.DataFrame(gate_rows)

    # ── write workbook ────────────────────────────────────────────────────────
    out = cfg.REPORTS_DIR_V11_1
    out.mkdir(parents=True, exist_ok=True)
    xlsx = out / "V11.1_ALT_fleet_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        risk.to_excel(xw, sheet_name="Risk_Ranking", index=False)
        rul_band.to_excel(xw, sheet_name="RUL_Band", index=False)
        cov_with_note.to_excel(xw, sheet_name="Covariates", index=False)
        variants_df.to_excel(xw, sheet_name="Variant_Comparison", index=False)
        emg_out.to_excel(xw, sheet_name="Emergency", index=False)
        ver_df.to_excel(xw, sheet_name="Verification", index=False)

    print(f"[excel_report] Saved {xlsx.name} ({xlsx.stat().st_size // 1024} KB, 7 sheets)")


if __name__ == "__main__":
    main()
