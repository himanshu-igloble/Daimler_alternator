import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from V11_2_ALT_rul_evidence_stack import (zscore_vs_baseline, first_breach_index, charging_bins,
                                          zone_of_m5, compute_m5)
import pandas as pd


def test_zscore_up_is_worse_noninverted():
    v = np.array([10., 10., 10., 20., 30.]); base = np.array([True, True, True, False, False])
    z = zscore_vs_baseline(v, base, invert=False)
    assert z[-1] > z[0]


def test_zscore_inverted_drop_is_worse():
    v = np.array([28., 28., 28., 26., 24.]); base = np.array([True, True, True, False, False])
    z = zscore_vs_baseline(v, base, invert=True)
    assert z[-1] > z[0]


def test_first_breach_index():
    z = np.array([0.1, 0.5, 1.0, 2.3, 2.9])
    assert first_breach_index(z, threshold=2.0) == 3
    assert first_breach_index(np.array([0., 1., 1.9]), threshold=2.0) is None


def test_charging_bins_ceiling():
    rng = np.random.default_rng(0)
    rpm = np.concatenate([np.full(40, 700.), np.full(40, 2000.)])
    vsi = np.concatenate([np.full(40, 25.), np.full(40, 28.)])
    centers, med = charging_bins(rpm, vsi, 600, 2500, 100)
    assert med[np.argmin(np.abs(centers - 2000))] > med[np.argmin(np.abs(centers - 700))]


def test_zone_of_m5_thresholds():
    assert zone_of_m5(0.05) == "GREEN"
    assert zone_of_m5(0.25) == "YELLOW"
    assert zone_of_m5(0.45) == "ORANGE"
    assert zone_of_m5(0.70) == "RED"


def test_compute_m5_range_and_ged_driver():
    n = 60
    d = pd.DataFrame(dict(vsi_mean=np.full(n, 28.2), vsi_std=np.zeros(n),
                          ged2_frac=np.zeros(n), vsi_range=np.zeros(n)))
    m5, contrib = compute_m5(d)
    assert (m5 >= 0).all() and (m5 <= 1).all()
    assert abs(m5[30]) < 1e-6                      # nominal/healthy -> ~0
    d2 = d.copy(); d2["ged2_frac"] = 0.10          # raise excitation-fault rate
    m5b, _ = compute_m5(d2)
    assert m5b[30] > m5[30]                         # GED2 rate drives score up


def test_debounce_smooths_short_excursions():
    import V11_2_ALT_rul_evidence_stack_effective as X
    dates = pd.date_range("2024-01-01", periods=60, freq="D").values
    lv = np.zeros(60, int)
    lv[10:13] = 1                                  # 3-day YELLOW spike -> noise
    lv[30:55] = 1                                  # 25-day YELLOW -> sustained
    out = X._debounce_levels(dates, lv, min_dwell_days=14)
    assert out[11] == 0                            # short spike smoothed back to GREEN
    assert out[40] == 1                            # sustained run adopted as YELLOW
    assert out[59] == 1                            # carries until a sustained better run


def test_showcase_detection_table_counts():
    p = os.path.join(os.path.dirname(__file__), "..", "results", "showcase", "detection_table.csv")
    df = pd.read_csv(p)
    f = df[df.failed]
    assert len(f) == 10                                       # 10 failed trucks
    assert int(f.alert.sum()) == 9                            # 9/10 in the alert zone (VIN5 missed)
    assert int(((df.band == "red") & df.failed).sum()) == 7   # 7 failures in red
    assert int(((df.band == "red") & ~df.failed).sum()) == 0  # red band is pure (0 healthy)
    assert int((f.hard_lead > 0).sum()) == 4                  # 4 give a hard lead
    assert int((f["mode"] == "abrupt").sum()) == 6            # 6/10 abrupt (no telemetry warning)


def test_cond_alert_deadband():
    import V11_2_ALT_rul_evidence_stack_effective as X
    assert X._cond_alert_level(0.16) == 0     # within deadband of 0.15 -> still GREEN (act-on)
    assert X._cond_alert_level(0.18) == 1     # clears 0.17 -> YELLOW
    assert X._cond_alert_level(0.40) == 2     # ORANGE
    assert X._cond_alert_level(0.60) == 3     # RED


def test_debounce_short_recovery_does_not_switch():
    import V11_2_ALT_rul_evidence_stack_effective as X
    dates = pd.date_range("2024-01-01", periods=40, freq="D").values
    lv = np.ones(40, int)                          # YELLOW throughout...
    lv[20:25] = 0                                  # ...except a 5-day GREEN dip (too short)
    out = X._debounce_levels(dates, lv, min_dwell_days=14)
    assert (out == 1).all()                        # the brief recovery is debounced away


def test_snapshot_columns_chronological():
    import V11_2_ALT_rul_evidence_stack_snapshot as S
    snap = S.compute_snapshot_for("VIN5_F_ALT")   # silent failure: only Early + Failure occur
    labels = [c[0] for c in snap["cols"]]
    assert labels == ["Early", "Failure"]          # absent-event columns are dropped
    dts = [c[1] for c in snap["cols"]]
    assert all(dts[i] <= dts[i + 1] for i in range(len(dts) - 1))   # chronological order
