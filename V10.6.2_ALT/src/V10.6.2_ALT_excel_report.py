"""
V10.6.2 Alternator — Excel workbook  (plan W5b)
================================================
Compact multi-sheet workbook of the validated outputs.  Reads median from
`median_ttf_days` (B1-safe) and verification gates by NAME (B2-safe).

Output: reports/V10.6.2_ALT_fleet_report.xlsx
"""
from __future__ import annotations

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


def main() -> None:
    wb = json.loads((pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json").read_text())
    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / "fleet_window.json").read_text())
    bt = json.loads((pathlib.Path(cfg.BACKTEST_CACHE) / "backtest_results.json").read_text())
    pred = pd.read_csv(pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_rul_predictions.csv")
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")
    btf = pd.read_csv(pathlib.Path(cfg.BACKTEST_CACHE) / "per_fold_residuals.csv")
    ver_path = pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_verification.json"
    ver = json.loads(ver_path.read_text()) if ver_path.exists() else {"gates": []}

    rew = bt["time_rewound"]["overall"]; tot = bt["lovo_total_ttf"]
    summary = pd.DataFrame({
        "metric": ["classifier_AUROC (frozen)", "fleet_median_TTF_empirical_days",
                   "fleet_window_p25_p75_days", "fleet_median_km_est", "fleet_median_ehrs_est",
                   "survival_median_context_days", "survival_shape_posterior",
                   "backtest_verdict", "rewound_MAE_model_days", "rewound_MAE_dummy_days",
                   "PI_coverage_total", "GED_sensitivity", "GED_false_alarms"],
        "value": ["0.927", fw["median_ttf_days"],
                  f"{fw['p25_ttf_days']:.0f}-{fw['p75_ttf_days']:.0f}",
                  fw["median_ttf_km_est"], fw["median_ttf_ehrs_est"],
                  wb["median_ttf_days"], wb["shape"], bt["time_rewound"]["verdict"],
                  rew["mae_model"], rew["mae_dummyA"], tot["pi_coverage_k_of_n"],
                  f"{int(ged[(ged.failed_flag==1)&(ged.ever_fired)].shape[0])}/10",
                  f"{int(ged[(ged.failed_flag==0)&(ged.ever_fired)].shape[0])}/15"],
    })
    risk = pred[pred.failed_flag == 0].sort_values("ridge_prob", ascending=False)[
        ["vin_label", "ridge_prob", "ridge_band", "risk_tier", "recommendation"]]
    rul_band = pred[["vin_label", "failed_flag", "current_age_days", "median_rul_days",
                     "rul_p10_days", "rul_p90_days", "risk_tier", "timing_confidence"]]
    ver_df = pd.DataFrame([{"gate": g["name"], "status": g["status"],
                            "type": "gating" if g["gating"] else "documented",
                            "details": g["details"]} for g in ver["gates"]])

    out = pathlib.Path(cfg.REPORTS_DIR_V2); out.mkdir(parents=True, exist_ok=True)
    xlsx = out / "V10.6.2_ALT_fleet_report.xlsx"
    with pd.ExcelWriter(xlsx) as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        risk.to_excel(xw, sheet_name="Risk_Ranking", index=False)
        rul_band.to_excel(xw, sheet_name="RUL_Band", index=False)
        ged.to_excel(xw, sheet_name="GED_Emergency", index=False)
        btf.to_excel(xw, sheet_name="Backtest_Folds", index=False)
        if len(ver_df):
            ver_df.to_excel(xw, sheet_name="Verification", index=False)
    print(f"[excel_report] Saved {xlsx.name}")


if __name__ == "__main__":
    main()
