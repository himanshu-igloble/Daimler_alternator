"""Phase 4: leave-one-VIN-out Ridge trial reproducing the frozen 0.9267 protocol.

Join-key note (2026-06-26):
  - Baseline CSV (V5.2_20_5_ALT_selected_features.csv):
      truck_id = VINx_F_ALT / VINx_NF_ALT  ← matches candidate VIN column
      VIN      = VIN1 / VIN2 / …            ← does NOT match candidate
  - Candidate CSV (4_ged_features.csv):
      VIN      = VINx_F_ALT / VINx_NF_ALT
  Fix: rename baseline `truck_id` → `VIN` before merge.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_SRC = pathlib.Path(__file__).resolve().parent


def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = _load("ged_common")

ALPHA = 1.0
FAMILY_A = [
    "vsi_std_ratio_30d",
    "vsi_dominant_freq",
    "vsi_spectral_entropy",
    "bat_charge_delta_trend_right",
    "vsi_range_trend_last30d",
    "progressive_drift",
]


def lovo_auroc(
    df: pd.DataFrame,
    feats,
    label_col: str = "failed",
    vin_col: str = "VIN",
) -> float:
    """Leave-one-VIN-out Ridge AUROC on POOLED out-of-fold probabilities.

    Protocol (frozen):
    - NaN imputed with TRAIN-only column median.
    - StandardScaler fit on train only.
    - RidgeClassifier(alpha=1.0) → decision_function → sigmoid → prob.
    - roc_auc_score on pooled predictions.
    """
    y = df[label_col].to_numpy().astype(int)
    probs = np.zeros(len(df))
    idx = np.arange(len(df))

    for i in range(len(df)):
        tr = idx != i
        Xtr = df.iloc[tr][feats].to_numpy(dtype=float)
        Xte = df.iloc[[i]][feats].to_numpy(dtype=float)

        # impute with train-only median
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)

        sc = StandardScaler().fit(Xtr)
        clf = RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr), y[tr])
        d = clf.decision_function(sc.transform(Xte))[0]
        probs[i] = 1.0 / (1.0 + np.exp(-d))

    return float(roc_auc_score(y, probs))


def main():
    base = pd.read_csv(
        C.ROOT / "V5.2_ALT" / "features" / "V5.2_20_5_ALT_selected_features.csv"
    )
    cand = pd.read_csv(C.RESULTS / "4_ged_features.csv")

    # --- join-key fix ---
    # baseline `VIN` col holds short labels (VIN1, VIN2, …) which do NOT match
    # candidate's VINx_F/NF_ALT labels.  `truck_id` holds the full labels.
    if "VIN" not in base.columns or not base["VIN"].astype(str).str.contains("_ALT").any():
        alt_found = False
        for col in ("truck_id", "VIN_LABEL", "vin"):
            if col in base.columns and base[col].astype(str).str.contains("_ALT").any():
                # drop any existing short-label VIN column to avoid duplicate
                if "VIN" in base.columns:
                    base = base.drop(columns=["VIN"])
                base = base.rename(columns={col: "VIN"})
                alt_found = True
                print(f"[join] renamed baseline '{col}' -> 'VIN' to match candidate labels")
                break
        if not alt_found:
            raise KeyError(
                "Cannot find a column in baseline CSV with VINx_*_ALT labels. "
                f"Columns are: {base.columns.tolist()}"
            )

    # merge
    df = base.merge(cand, on="VIN", how="left")

    # --- diagnostics ---
    print(f"[join] baseline cols  : {base.columns.tolist()}")
    print(f"[join] candidate cols : {cand.columns.tolist()}")
    print(f"[join] merged rows    : {len(df)}  (expect 25)")
    print(f"[join] FAMILY_A nulls : {df[FAMILY_A].isna().sum().sum()}")

    assert len(df) == 25, f"Expected 25 rows after merge, got {len(df)}"
    # Note: progressive_drift has 1 native NaN in the baseline CSV (VIN3_NF_ALT).
    # The LOVO loop imputes NaN with train-only column median, so this is expected.
    # We only assert no NEW nulls were introduced by the join (candidate columns
    # should be fully populated, baseline nulls remain at their source count).
    base_nulls = base[FAMILY_A].isna().sum().sum()
    merged_nulls = df[FAMILY_A].isna().sum().sum()
    assert merged_nulls == base_nulls, (
        f"Merge introduced {merged_nulls - base_nulls} new NaN(s) in FAMILY_A "
        f"(baseline had {base_nulls}, merged has {merged_nulls}) -- join mismatch"
    )
    if base_nulls > 0:
        null_detail = df[FAMILY_A].isna().sum()
        print(f"[join] FAMILY_A source NaN (will be imputed in LOVO loop): {null_detail[null_detail > 0].to_dict()}")

    # --- baseline gate ---
    a0 = lovo_auroc(df, FAMILY_A)
    print(f"\nBASELINE FAMILY_A AUROC = {a0:.4f} (expect ~0.9267)")
    assert abs(a0 - 0.9267) < 0.005, (
        f"Baseline not reproduced (got {a0:.4f}) — fix join/protocol before testing candidates"
    )

    # --- candidate trials ---
    new_feats = [
        "ged2_acceleration",
        "ged2_onset_slope",
        "ged2_rate_idle",
        "ged2_rate_cruise",
        "resid_mean",
        "resid_slope_30d",
        "resid_oscillation",
    ]

    rows = [{"set": "FAMILY_A", "features": "baseline-6", "auroc": a0, "delta": 0.0}]

    for f in new_feats:
        a = lovo_auroc(df, FAMILY_A + [f])
        rows.append({"set": f"FAMILY_A+{f}", "features": f, "auroc": a, "delta": a - a0})
        print(f"  FAMILY_A + {f:<28s} AUROC = {a:.4f}  d = {a - a0:+.4f}")

    a_all = lovo_auroc(df, FAMILY_A + new_feats)
    rows.append(
        {
            "set": "+ALL_GED",
            "features": ";".join(new_feats),
            "auroc": a_all,
            "delta": a_all - a0,
        }
    )
    print(f"  FAMILY_A + ALL_GED                       AUROC = {a_all:.4f}  d = {a_all - a0:+.4f}")

    out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    out.to_csv(C.RESULTS / "4_lovo_trial.csv", index=False)
    print("\n--- Ranked results ---")
    print(out.to_string(index=False))
    print(f"\nOutput: {C.RESULTS / '4_lovo_trial.csv'}")


if __name__ == "__main__":
    main()
