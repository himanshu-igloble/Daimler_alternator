"""Task 5 — Crank-recovery TREND from V11_ALT_heuristics forensic daily panels.

Checks for crank_recovery_t column in per-VIN daily CSVs.
If found: compute self-normalized 60-day slope, Mann-Whitney AUROC.
Writes: V12_ALT_GED/results/r3_crank_trend.json
"""
import sys
import json
import pathlib

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS   = pathlib.Path(__file__).resolve().parents[1] / "results"
FORENSICS = pathlib.Path(__file__).resolve().parents[2] / "V11_ALT_heuristics" / "cache" / "forensics"

FAILED_VINS    = [f"VIN{i}_F_ALT" for i in range(1, 11)]
NONFAILED_VINS = [f"VIN{i}_NF_ALT" for i in range(1, 16)]
ALL_VINS       = FAILED_VINS + NONFAILED_VINS

TARGET_COL = "crank_recovery_t"

print("=" * 70)
print("TASK 5 — CRANK-RECOVERY TREND")
print("=" * 70)

# ── check if column exists ────────────────────────────────────────────────────
sample_path = FORENSICS / "VIN1_F_ALT_daily.csv"
if not sample_path.exists():
    result = {"status": "SKIP", "reason": "Forensic daily CSVs not found at expected path"}
    print(f"SKIP: {sample_path} not found")
else:
    sample_df = pd.read_csv(sample_path, nrows=2)
    if TARGET_COL not in sample_df.columns:
        # list available columns for the record
        result = {
            "status": "SKIP",
            "reason": f"Column '{TARGET_COL}' not found in forensic daily CSVs",
            "available_columns": sample_df.columns.tolist(),
        }
        print(f"SKIP: '{TARGET_COL}' not in columns.")
        print(f"Available columns: {sample_df.columns.tolist()}")
    else:
        print(f"Column '{TARGET_COL}' found. Proceeding.")

        # ── per-VIN: load, compute self-normalized 60d slope ─────────────────
        vin_features = []
        for vin in ALL_VINS:
            fpath = FORENSICS / f"{vin}_daily.csv"
            if not fpath.exists():
                print(f"  {vin}: file not found, skip")
                continue

            df = pd.read_csv(fpath)
            if TARGET_COL not in df.columns:
                print(f"  {vin}: column absent")
                continue
            if "dtf" not in df.columns:
                # try vin_label-based lookup
                df = df.sort_values("day").reset_index(drop=True) if "day" in df.columns else df

            series = df[TARGET_COL].dropna().values
            failed = vin.endswith("_F_ALT")

            if len(series) < 10:
                print(f"  {vin}: too few points ({len(series)})")
                continue

            # baseline = all but last 60
            if len(series) > 60:
                base_vals  = series[:-60]
                last60     = series[-60:]
            else:
                base_vals  = series
                last60     = series

            base_mean  = float(np.mean(base_vals))
            base_std   = float(np.std(base_vals)) + 1e-9

            # slope of last-60 (simple linear regression index vs value)
            x    = np.arange(len(last60))
            if len(last60) >= 2:
                slope = float(np.polyfit(x, last60, 1)[0])
            else:
                slope = 0.0

            # self-normalize: slope in units of baseline stds per day
            norm_slope = slope / base_std

            vin_features.append({
                "vin":        vin,
                "failed":     failed,
                "n_obs":      int(len(series)),
                "base_mean":  round(base_mean, 5),
                "base_std":   round(base_std - 1e-9, 5),
                "last60_slope":      round(slope, 6),
                "norm_slope":        round(norm_slope, 6),
            })

        if len(vin_features) < 10:
            result = {
                "status": "SKIP",
                "reason": f"Too few usable VINs ({len(vin_features)}) for meaningful test",
                "per_vin": vin_features,
            }
            print(f"SKIP: only {len(vin_features)} usable VINs.")
        else:
            feat_df = pd.DataFrame(vin_features)

            # ── Mann-Whitney AUROC (model-free) ──────────────────────────────
            from scipy.stats import mannwhitneyu
            f_vals  = feat_df.loc[feat_df["failed"] == True,  "norm_slope"].values
            nf_vals = feat_df.loc[feat_df["failed"] == False, "norm_slope"].values

            if len(f_vals) == 0 or len(nf_vals) == 0:
                mw_auroc = None
                mw_pval  = None
            else:
                stat, pval = mannwhitneyu(f_vals, nf_vals, alternative="two-sided")
                mw_auroc = float(stat / (len(f_vals) * len(nf_vals)))
                mw_pval  = float(pval)

            # ── GED-silent failed VINs late rise ─────────────────────────────
            # GED-silent = F VINs other than VIN1 and VIN10
            ged_silent = [v for v in ["VIN2_F_ALT","VIN3_F_ALT","VIN4_F_ALT","VIN5_F_ALT",
                                       "VIN6_F_ALT","VIN7_F_ALT","VIN8_F_ALT","VIN9_F_ALT"]
                           if v in feat_df["vin"].values]
            ged_silent_slopes = feat_df[feat_df["vin"].isin(ged_silent)]["norm_slope"].tolist()

            print(f"\nMann-Whitney AUROC (norm_slope, failed vs NF): {mw_auroc:.4f}  p={mw_pval:.4f}" if mw_auroc else "Mann-Whitney: N/A")
            print(f"GED-silent F VINs norm_slope: {ged_silent_slopes}")
            print()
            print(feat_df[["vin","failed","n_obs","norm_slope"]].to_string(index=False))

            result = {
                "status":              "COMPLETE",
                "target_col":          TARGET_COL,
                "n_vins":              len(vin_features),
                "mw_auroc":            mw_auroc,
                "mw_pval":             mw_pval,
                "ged_silent_f_vins":   ged_silent,
                "ged_silent_slopes":   ged_silent_slopes,
                "per_vin":             vin_features,
            }

out_path = RESULTS / "r3_crank_trend.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=str)
print()
print(f"JSON written: {out_path}")
print("=" * 70)
