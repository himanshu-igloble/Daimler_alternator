"""
V11.1_ALT — Master Orchestrator
================================
Runs the V11.1_ALT pipeline in dependency order.  verify runs LAST so it
can audit all generated cache and results (five honest gates).

NOTE: The full run takes approximately 30 minutes due to the LOVO backtest
(10-fold × 3 variants × grid posterior fits + 2000 draws each).  Do NOT
run this interactively during development — use unit tests and standalone
verify instead.

Stage order:
  1. V11_1_ALT_covariates      — leakage-safe covariate vectors (x1, x2)
  2. V11_1_ALT_weibull_fleet   — fleet AFT posterior for M0/M1/M2
  3. V11_1_ALT_backtest        — time-rewound LOVO backtest + variant selection
  4. V11_1_ALT_predictive_rul  — per-VIN RUL bands from chosen posterior
  5. V11_1_ALT_emergency       — GED/exceed/compound emergency channels
  6. V11_1_ALT_decisions       — 2x2 risk × time decision engine
  7. V11_1_ALT_assemble_rul    — final RUL table + empirical fleet window
  8. V11_1_ALT_narrative_rul   — per-VIN narrative summaries
  9. V11_1_ALT_verify          — five honest gates; exits 1 on any failure

Graphs and customer-facing reports are built by later stages (not yet
wired into this orchestrator).

GATE_FAIL: If verify exits with code != 0, the orchestrator records
GATE_FAIL and exits with code 1.  No subsequent stage runs.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
import traceback

_src = pathlib.Path(__file__).resolve().parent

# Load config for version banner
_cfg_spec = importlib.util.spec_from_file_location(
    "V11_1_ALT_config", str(_src / "V11_1_ALT_config.py"))
cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(cfg)

SCRIPTS = [
    "V11_1_ALT_covariates",
    "V11_1_ALT_weibull_fleet",
    "V11_1_ALT_backtest",
    "V11_1_ALT_predictive_rul",
    "V11_1_ALT_emergency",
    "V11_1_ALT_decisions",
    "V11_1_ALT_assemble_rul",
    "V11_1_ALT_narrative_rul",
    "V11_1_ALT_verify",
]


def _run(name):
    path = _src / f"{name}.py"
    if not path.exists():
        return "SKIP", 0.0, "missing"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return "OK", time.perf_counter() - t0, None
    except SystemExit as e:
        # verify raises SystemExit(1) on gating failure — capture, report, halt
        return (
            "OK" if (e.code in (0, None)) else "GATE_FAIL",
            time.perf_counter() - t0,
            f"exit={e.code}",
        )
    except Exception as e:
        print(f"\n{'!' * 70}\nERROR in {name}:\n{traceback.format_exc()}{'!' * 70}\n")
        return "FAIL", time.perf_counter() - t0, str(e)


def main():
    print("=" * 70)
    print(f"V11.1_ALT ALTERNATOR — ORCHESTRATOR  ({cfg.VERSION})")
    print("=" * 70)

    results = []
    for i, name in enumerate(SCRIPTS, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(SCRIPTS)}] {name}\n{'=' * 70}")
        status, elapsed, err = _run(name)
        results.append((name, status, elapsed, err))
        print(
            f"\n  >> {name}: {status} ({elapsed:.1f}s)"
            f"{'' if not err else ' - ' + str(err)}"
        )
        # Halt immediately on gate failure
        if status == "GATE_FAIL":
            print(f"\n  !! GATE_FAIL in {name} — stopping pipeline")
            break

    print(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    nok = nfail = 0
    for idx, (n, s, el, err) in enumerate(results, 1):
        print(f"{idx:<3} {n:<34} {s:<10} {el:>6.1f}s  {err or ''}")
        nok += s == "OK"
        nfail += s in ("FAIL", "GATE_FAIL")
    print(f"\n{nok}/{len(SCRIPTS)} OK, {nfail} failed/gate-fail")
    if nfail:
        sys.exit(1)


if __name__ == "__main__":
    main()
