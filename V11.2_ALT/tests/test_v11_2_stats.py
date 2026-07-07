import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from V11_2_ALT_common import auroc_from_scores, cliffs_delta, concordant_pairs, jcopendate_failure_age

def test_auroc_perfect_separation():
    assert abs(auroc_from_scores([0.9, 0.8, 0.7], [0.3, 0.2, 0.1]) - 1.0) < 1e-9

def test_auroc_reversed():
    assert abs(auroc_from_scores([0.1, 0.2], [0.8, 0.9]) - 0.0) < 1e-9

def test_auroc_ties_count_half():
    assert abs(auroc_from_scores([0.5], [0.5]) - 0.5) < 1e-9

def test_concordant_pairs_counts():
    c, t, d, n = concordant_pairs([0.9, 0.5], [0.4, 0.5])
    assert (c, t, d, n) == (3, 1, 0, 4)

def test_cliffs_delta_matches_auroc():
    f, h = [0.9, 0.8, 0.7], [0.3, 0.2, 0.1]
    assert abs(cliffs_delta(f, h) - (2*auroc_from_scores(f, h) - 1)) < 1e-9

def test_jcopendate_failure_age():
    import datetime as dt
    t0 = dt.date(2024, 1, 1)
    jc = dt.date(2025, 11, 29)
    assert jcopendate_failure_age(t0, jc) == (jc - t0).days

def test_clip_rul_at_failure_age():
    import numpy as np
    ages = np.array([0, 100, 200, 300, 400])
    rul  = np.array([400, 300, 200, 100, 50], float)
    fa = 250
    clipped = np.where(ages >= fa, 0.0, rul)
    assert clipped[ages >= fa].sum() == 0
    assert (ages <= fa).sum() == 3
