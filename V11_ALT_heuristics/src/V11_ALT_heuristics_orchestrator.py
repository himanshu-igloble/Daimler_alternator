"""V11_ALT_heuristics — pipeline orchestrator.

Runs the precursor channel end to end:
  forensic -> changepoint -> compound -> verify (hard gate) -> compare.
The frozen classifier / Weibull are reused by reference and not re-run.
"""
from __future__ import annotations

import importlib.util
import pathlib
import time

_src = pathlib.Path(__file__).resolve().parent
SCRIPTS = [
    "V11_ALT_heuristics_forensic",
    "V11_ALT_heuristics_changepoint",
    "V11_ALT_heuristics_compound",
    "V11_ALT_heuristics_verify",
    "V11_ALT_heuristics_compare",
]


def _run(name):
    path = _src / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return "OK", time.perf_counter() - t0, None
    except SystemExit as e:
        ok = e.code in (0, None)
        return ("OK" if ok else "GATE_FAIL"), time.perf_counter() - t0, f"exit={e.code}"


def main():
    print("=" * 70)
    print("V11_ALT_heuristics pipeline")
    print("=" * 70)
    results = []
    for name in SCRIPTS:
        print(f"\n>>> {name}")
        status, dt, note = _run(name)
        results.append((name, status, dt, note))
        print(f"<<< {name}: {status} ({dt:.1f}s){'' if not note else ' ' + note}")
        if status == "GATE_FAIL":
            print("\nPipeline halted: a hard gate failed.")
            break
    print("\n" + "=" * 70)
    for name, status, dt, note in results:
        print(f"  {status:<10} {name:<36} {dt:6.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
