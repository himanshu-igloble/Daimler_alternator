"""V11_ALT_heuristics — honest verification gates.

G2 forensic artifacts exist.
G3 recall does not regress below V10.6.2 (V11 superset must be >=).
G4 W6 forbidden-pattern scan: no re-fitting of the frozen classifier in src/.
Exits non-zero on any hard-gate failure so the orchestrator surfaces it.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_src / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_ALT_heuristics_config")
FORENSICS = cfg.FORENSICS


def scan_forbidden(text: str):
    """Return forbidden classifier-refit patterns present in text."""
    return [p for p in cfg.FORBIDDEN_FIT_PATTERNS if p in text]


def main() -> None:
    failures = []

    required = ["earliest_signal_per_vin.csv", "nf_baseline.csv",
                "failed_window_deviations.csv", "nf_self_test.csv",
                "changepoint_per_vin.csv", "compound_alarm_lovo.csv"]
    for fn in required:
        if not (FORENSICS / fn).exists():
            failures.append(f"G2 missing artifact: {fn}")

    if (FORENSICS / "earliest_signal_per_vin.csv").exists():
        v11 = pd.read_csv(FORENSICS / "earliest_signal_per_vin.csv")
        v1062 = pd.read_csv(cfg.V1062_FORENSICS / "earliest_signal_per_vin.csv")
        n11 = int((v11["verdict"] == "discriminative_precursor").sum())
        n1062 = int((v1062["verdict"] == "discriminative_precursor").sum())
        print(f"  G3 recall: V11 {n11}/10  vs  V10.6.2 {n1062}/10")
        if n11 < n1062:
            failures.append(f"G3 recall regression: V11 {n11} < V10.6.2 {n1062}")

    for py in sorted(_src.glob("V11_ALT_heuristics_*.py")):
        if py.name.endswith("config.py"):
            continue
        hits = scan_forbidden(py.read_text(encoding="utf-8"))
        if hits:
            failures.append(f"G4 forbidden pattern {hits} in {py.name}")

    if failures:
        print("[v11 verify] FAIL:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("[v11 verify] PASS — all honest gates green")


if __name__ == "__main__":
    main()
