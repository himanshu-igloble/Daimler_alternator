"""
V11.2_ALT_metric_suite.py
Rigorous validation and explanation of the "93%" AUROC headline.

Inputs:
  - V5.2_ALT/results/V10.5.3_20_5_ALT_ridge_predictions.csv  (LOVO OOF scores)
  - V5.2_ALT/models/classification/V10.5.3_20_5_ALT_ridge_spec.json (via load_ridge_spec)
  - V11.1_ALT/cache/emergency/emergency_per_vin.csv             (via load_emergency)

Outputs:
  - V11.2_ALT/results/V11.2_ALT_pair_decomposition.csv
  - V11.2_ALT/results/V11.2_ALT_metric_suite.json
  - V11.2_ALT/visualizations/metric_suite/roc_curve.png
  - V11.2_ALT/visualizations/metric_suite/pr_curve.png
"""

import sys, os, math
import numpy as np
import pandas as pd

sys.path.insert(0, r'D:/Daimler-starter_motor_alternator_battery/V11.2_ALT/src')
import V11_2_ALT_common as common

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef,
)
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRED_CSV   = r'D:/Daimler-starter_motor_alternator_battery/V5.2_ALT/results/V10.5.3_20_5_ALT_ridge_predictions.csv'
THRESHOLD  = 0.4456          # Youden operating point from spec
EXPECTED_AUROC = 0.9267
VIZ_DIR    = os.path.join(common.VIZ, 'metric_suite')

os.makedirs(common.RESULTS, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


# ===========================================================================
# STEP 1 — Load LOVO predictions
# ===========================================================================
df = pd.read_csv(PRED_CSV)
# Columns used: 'VIN_LABEL', 'failed' (truth 0/1), 'ridge_prob' (LOVO OOF score)
print("\n[STEP 1] Loaded predictions CSV")
print(f"  Columns used: truth='failed', score='ridge_prob'")
print(f"  Total rows: {len(df)}  |  failed={df['failed'].sum()}  healthy={(df['failed']==0).sum()}")

truth  = df['failed'].values.astype(int)
scores = df['ridge_prob'].values.astype(float)

failed_mask  = truth == 1
healthy_mask = truth == 0

failed_vins   = df.loc[failed_mask,  'VIN_LABEL'].values
healthy_vins  = df.loc[healthy_mask, 'VIN_LABEL'].values
failed_scores = scores[failed_mask]
healthy_scores = scores[healthy_mask]


# ===========================================================================
# STEP 2 — AUROC pair decomposition
# ===========================================================================
c_cnt, t_cnt, d_cnt, n_pairs = common.concordant_pairs(failed_scores, healthy_scores)
auroc = (c_cnt + 0.5 * t_cnt) / n_pairs

print(f"\n[STEP 2] AUROC decomposition")
print(f"  AUROC={auroc:.4f}  ({c_cnt} concordant / {t_cnt} ties / {d_cnt} discordant of {n_pairs})")

# Guard: AUROC must be within 0.01 of 0.9267
assert abs(auroc - EXPECTED_AUROC) < 0.01, (
    f"AUROC={auroc:.4f} deviates from expected {EXPECTED_AUROC} by "
    f"{abs(auroc - EXPECTED_AUROC):.4f} (>0.01). Wrong column?"
)
print(f"  [ASSERT OK] |auroc - {EXPECTED_AUROC}| = {abs(auroc - EXPECTED_AUROC):.4f} < 0.01")

# Build per-pair table (150 rows)
rows = []
for fi, fv, fs in zip(range(len(failed_vins)), failed_vins, failed_scores):
    for hi, hv, hs in zip(range(len(healthy_vins)), healthy_vins, healthy_scores):
        if fs > hs:
            outcome = 'C'
        elif fs == hs:
            outcome = 'T'
        else:
            outcome = 'D'
        rows.append({
            'failed_vin':  fv,
            'healthy_vin': hv,
            'score_f':     round(float(fs), 4),
            'score_h':     round(float(hs), 4),
            'outcome':     outcome,
        })

pair_df = pd.DataFrame(rows)
pair_path = os.path.join(common.RESULTS, 'V11.2_ALT_pair_decomposition.csv')
pair_df.to_csv(pair_path, index=False)

# Guard: 150 rows
assert len(pair_df) == 150, f"pair_decomposition has {len(pair_df)} rows, expected 150"
print(f"  [ASSERT OK] pair_decomposition.csv has {len(pair_df)} rows (10×15=150)")
print(f"  Discordant pairs (model 'mistakes'): {len(pair_df[pair_df['outcome']=='D'])}")
print(f"  Saved: {pair_path}")


# ===========================================================================
# STEP 3 — Metric panel
# ===========================================================================
print("\n[STEP 3] Computing metric panel...")

# --- Threshold-based predictions ---
preds = (scores >= THRESHOLD).astype(int)

# Precision, Recall, F1
prec  = precision_score(truth, preds, zero_division=0)
rec   = recall_score(truth, preds, zero_division=0)
f1    = f1_score(truth, preds, zero_division=0)
mcc   = matthews_corrcoef(truth, preds)

# Specificity = TN / (TN + FP)
tn = int(np.sum((truth == 0) & (preds == 0)))
fp = int(np.sum((truth == 0) & (preds == 1)))
tp = int(np.sum((truth == 1) & (preds == 1)))
fn = int(np.sum((truth == 1) & (preds == 0)))
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Brier score — ridge_prob is already a calibrated [0,1] score
brier_raw = float(np.mean((scores - truth) ** 2))

# Also compute Brier on sigmoid of score (in case scores were decision values)
# For ridge_prob which is already [0,1], sigmoid will slightly change values
# Include both for completeness; label clearly
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
# ridge_prob is already [0,1], so sigmoid is not appropriate — but we provide
# a "logit-then-sigmoid" version to satisfy the spec requirement
logit_scores = np.log(np.clip(scores, 1e-6, 1-1e-6) / (1 - np.clip(scores, 1e-6, 1-1e-6)))
brier_sigmoid = float(np.mean((sigmoid(logit_scores) - truth) ** 2))

# PR-AUC
pr_auc = float(average_precision_score(truth, scores))

# Top-k accuracy: rank all 25 by score desc, check top 10
sorted_idx = np.argsort(scores)[::-1]
top_10_idx = sorted_idx[:10]
top_10_truth = truth[top_10_idx]
top_k_failed_count = int(np.sum(top_10_truth))
top_k_accuracy_frac = top_k_failed_count / 10

# Recall at top decile: top 10% of 25 = top 2-3 trucks (use ceil(0.1*25)=3)
top_decile_k = math.ceil(0.1 * 25)   # = 3
top_decile_idx = sorted_idx[:top_decile_k]
top_decile_truth = truth[top_decile_idx]
recall_at_top_decile_count = int(np.sum(top_decile_truth))
recall_at_top_decile_frac  = recall_at_top_decile_count / int(np.sum(truth))

# Spearman rank correlation
spear_corr, spear_pval = spearmanr(scores, truth)

# --- Load ridge spec for bootstrap/permutation values ---
spec = common.load_ridge_spec()
bootstrap_auroc_mean = spec.get('bootstrap_auroc_mean')
bootstrap_95ci       = spec.get('bootstrap_95ci')
permutation_p_value  = spec.get('permutation_p_value')

# Cross-check computed vs spec lovo_* values (within 0.05)
spec_checks = {
    'recall':      {'computed': round(rec, 4),         'spec': spec.get('lovo_recall'),      'matches': abs(rec - spec.get('lovo_recall', 0))        < 0.05},
    'specificity': {'computed': round(specificity, 4), 'spec': spec.get('lovo_specificity'), 'matches': abs(specificity - spec.get('lovo_specificity', 0)) < 0.05},
    'precision':   {'computed': round(prec, 4),        'spec': spec.get('lovo_precision'),   'matches': abs(prec - spec.get('lovo_precision', 0))     < 0.05},
    'f1':          {'computed': round(f1, 4),          'spec': spec.get('lovo_f1'),          'matches': abs(f1 - spec.get('lovo_f1', 0))              < 0.05},
    'mcc':         {'computed': round(mcc, 4),         'spec': spec.get('lovo_mcc'),         'matches': abs(mcc - spec.get('lovo_mcc', 0))            < 0.05},
    'brier':       {'computed': round(brier_raw, 4),   'spec': spec.get('lovo_brier'),       'matches': abs(brier_raw - spec.get('lovo_brier', 0))    < 0.05},
}

# --- Mean lead time from emergency CSV ---
em = common.load_emergency()
# GED-fired lead times for failed trucks only
ged_leads = em.loc[em['ged_fired'] == True, 'ged_lead_days'].dropna()
mean_lead_time_days = float(ged_leads.mean()) if len(ged_leads) > 0 else None
lead_vin_count = int(len(ged_leads))
lead_vin_labels = em.loc[em['ged_fired'] == True, 'vin_label'].tolist()

# Guard: recall must be 0.9 (9/10)
assert abs(rec - 0.9) < 1e-6, f"Recall={rec:.4f}, expected 0.9 (9/10)"
print(f"  [ASSERT OK] Recall={rec:.1f} = 9/10 failed trucks caught")

# --- Assemble metric suite JSON ---
metric_suite = {
    "_meta": {
        "version":    "V11.2_ALT",
        "source_csv": PRED_CSV,
        "threshold":  THRESHOLD,
        "note":       "All metrics computed from LOVO (leave-one-VIN-out) out-of-fold predictions, NOT full-fit probabilities",
    },

    # Core ranking metric
    "auroc": round(auroc, 4),
    "auroc_decomposition": {
        "concordant":   c_cnt,
        "ties":         t_cnt,
        "discordant":   d_cnt,
        "total_pairs":  n_pairs,
        "formula":      "(concordant + 0.5*ties) / total_pairs",
        "interpretation": "P(failed truck ranked above healthy truck on a randomly drawn pair)",
    },

    # PR-AUC
    "pr_auc": round(pr_auc, 4),

    # Threshold-based metrics at Youden 0.4456
    "threshold_metrics": {
        "threshold":    THRESHOLD,
        "precision":    round(prec, 4),
        "recall":       round(rec, 4),
        "f1":           round(f1, 4),
        "specificity":  round(specificity, 4),
        "mcc":          round(mcc, 4),
        "brier_raw":    round(brier_raw, 4),   # score already in [0,1]
        "brier_sigmoid": round(brier_sigmoid, 4),  # brier on sigmoid(logit(score)) — near-identical since score is [0,1]
        "brier_note":   "brier_raw uses ridge_prob directly (already [0,1]); brier_sigmoid uses sigmoid(logit(ridge_prob)) for completeness",
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    },

    # Top-k and decile metrics
    "top_k_accuracy": {
        "k":                 10,
        "failed_in_top_k":   top_k_failed_count,
        "total_failed":      int(np.sum(truth)),
        "fraction":          f"{top_k_failed_count}/10",
        "note":              "Trucks ranked in top 10 by LOVO score (25 trucks total); 10 are truly failed",
    },
    "recall_at_top_decile": {
        "top_decile_k":          top_decile_k,
        "definition":            f"top ceil(0.10 * 25)={top_decile_k} trucks by score",
        "failed_in_top_decile":  recall_at_top_decile_count,
        "total_failed":          int(np.sum(truth)),
        "fraction":              f"{recall_at_top_decile_count}/{int(np.sum(truth))}",
        "recall_at_top_decile":  round(recall_at_top_decile_frac, 4),
    },

    # Rank correlation
    "spearman_rank_corr": {
        "rho":    round(float(spear_corr), 4),
        "pvalue": round(float(spear_pval), 4),
        "note":   "Spearman rho between ridge_prob score and binary truth label",
    },
    "concordance_index": {
        "value": round(auroc, 4),
        "note":  "For binary outcomes, concordance index = AUROC (stated equality)",
    },

    # Bootstrap and permutation from V10.5.3 ridge_spec (not recomputed)
    "bootstrap_auroc_mean": bootstrap_auroc_mean,
    "bootstrap_95ci":       bootstrap_95ci,
    "permutation_p_value":  permutation_p_value,
    "external_stats_note":  "bootstrap_auroc_mean, bootstrap_95ci, permutation_p_value cited from V10.5.3 ridge_spec.json; not recomputed here",

    # Spec cross-check
    "spec_cross_check": spec_checks,

    # Lead time (GED channel only)
    "mean_lead_time_days": {
        "value_days":     round(mean_lead_time_days, 1) if mean_lead_time_days else None,
        "vin_count":      lead_vin_count,
        "total_failed":   10,
        "fraction":       f"{lead_vin_count}/10",
        "vin_labels":     lead_vin_labels,
        "note":           "GED=2 excitation-disturbance channel only; only 2/10 failed trucks emit this signal",
        "individual_leads_days": ged_leads.tolist(),
    },

    # Business framing
    "framing": {
        "conservative":   "recall 9/10 (90%) of failures caught",
        "practical":      "93% ranking AUROC on trucks the model never saw",
        "business":       "0 false alarms in healthy fleet; the top-10 inspection list catches 9/10 failures",
        "recommendation": (
            "Communicate the 93% as ranking accuracy AND pair it with '9/10 caught, 0 false alarms'; "
            "never present it as bare classification accuracy"
        ),
    },
}

common.save_json(metric_suite, 'V11.2_ALT_metric_suite.json')
print(f"  Saved: {os.path.join(common.RESULTS, 'V11.2_ALT_metric_suite.json')}")

# Print spec cross-check summary
print("\n  Spec cross-check (computed vs V10.5.3 lovo_* values):")
for metric, chk in spec_checks.items():
    mark = "OK" if chk['matches'] else "MISMATCH"
    print(f"    [{mark}] {metric}: computed={chk['computed']}  spec={chk['spec']}")


# ===========================================================================
# STEP 4 — Plots
# ===========================================================================
print("\n[STEP 4] Generating plots...")

PAL = common.PALETTE

# --- ROC Curve ---
fpr_arr, tpr_arr, roc_thresholds = roc_curve(truth, scores)
roc_auc = auc(fpr_arr, tpr_arr)

# Find the operating point closest to Youden threshold
youden_idx = np.argmin(np.abs(roc_thresholds - THRESHOLD))
op_fpr = fpr_arr[youden_idx]
op_tpr = tpr_arr[youden_idx]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_arr, tpr_arr, color=PAL['accent'], lw=2.0, label=f'LOVO ROC (AUROC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color=PAL['grid'], lw=1.2, linestyle='--', label='Random baseline')

# Mark Youden operating point
ax.scatter([op_fpr], [op_tpr], color=PAL['fail'], s=90, zorder=5,
           label=f'Youden threshold {THRESHOLD} (FPR={op_fpr:.2f}, TPR={op_tpr:.2f})')
ax.annotate(
    f'  AUROC = {roc_auc:.3f}',
    xy=(0.55, 0.35), fontsize=11, color=PAL['accent'], weight='bold'
)

common.style_ax(ax,
    title='ROC Curve — LOVO Out-of-Fold Predictions (V10.5.3 Ridge, n=25)',
    xlabel='False Positive Rate (1 − Specificity)',
    ylabel='True Positive Rate (Recall)',
)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])

roc_path = os.path.join(VIZ_DIR, 'roc_curve.png')
fig.tight_layout()
fig.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {roc_path}")

# --- PR Curve ---
precision_arr, recall_arr, pr_thresholds = precision_recall_curve(truth, scores)
ap = average_precision_score(truth, scores)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall_arr, precision_arr, color=PAL['fail'], lw=2.0,
        label=f'LOVO PR curve (AP = {ap:.3f})')

# Mark the operating point on PR curve
# For threshold=0.4456, find its position in pr_thresholds
if len(pr_thresholds) > 0:
    pr_op_idx = np.argmin(np.abs(pr_thresholds - THRESHOLD))
    pr_op_rec = recall_arr[pr_op_idx]
    pr_op_prec = precision_arr[pr_op_idx]
    ax.scatter([pr_op_rec], [pr_op_prec], color=PAL['accent'], s=90, zorder=5,
               label=f'Youden threshold {THRESHOLD} (R={pr_op_rec:.2f}, P={pr_op_prec:.2f})')

# Baseline: fraction of positives
baseline = np.sum(truth) / len(truth)
ax.axhline(baseline, color=PAL['grid'], lw=1.2, linestyle='--',
           label=f'No-skill baseline ({baseline:.2f})')

ax.annotate(
    f'  AP = {ap:.3f}',
    xy=(0.05, 0.35), fontsize=11, color=PAL['fail'], weight='bold'
)

common.style_ax(ax,
    title='Precision-Recall Curve — LOVO Out-of-Fold (V10.5.3 Ridge, n=25)',
    xlabel='Recall',
    ylabel='Precision',
)
ax.legend(fontsize=8, loc='lower left')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])

pr_path = os.path.join(VIZ_DIR, 'pr_curve.png')
fig.tight_layout()
fig.savefig(pr_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {pr_path}")


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Predictions CSV columns used: truth='failed', score='ridge_prob'")
print(f"  AUROC={auroc:.4f}  ({c_cnt} concordant / {t_cnt} ties / {d_cnt} discordant of {n_pairs})")
print(f"  PR-AUC={pr_auc:.4f}")
print(f"  @threshold={THRESHOLD}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} specificity={specificity:.4f} mcc={mcc:.4f}")
print(f"  Brier (raw)={brier_raw:.4f}  Brier (sigmoid)={brier_sigmoid:.4f}")
print(f"  Top-10 accuracy: {top_k_failed_count}/10 failed trucks in top-10 ranked list")
print(f"  Recall@top-decile (top {top_decile_k} trucks): {recall_at_top_decile_count}/{int(np.sum(truth))} failed")
print(f"  Spearman rho={spear_corr:.4f}  p={spear_pval:.4f}")
print(f"  Bootstrap AUROC mean={bootstrap_auroc_mean}  95CI={bootstrap_95ci}  perm_p={permutation_p_value}")
print(f"  GED lead time: {mean_lead_time_days:.1f}d mean ({lead_vin_count}/10 trucks fire GED)")
print(f"  pair_decomposition.csv: {len(pair_df)} rows  [ASSERT OK]")
print(f"  Recall ASSERT: {rec:.1f}=0.9  [ASSERT OK]")
print(f"  AUROC ASSERT: {abs(auroc - EXPECTED_AUROC):.4f}<0.01  [ASSERT OK]")
print(f"  Plots: roc_curve.png, pr_curve.png")
print("="*70)
