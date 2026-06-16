import importlib.util
import pathlib
import pandas as pd

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CMP = _load("V11_ALT_heuristics_compare")


def test_classify_feature():
    assert CMP.classify_feature(failed_hits=3, nf_false=0) == "generalizes"
    assert CMP.classify_feature(failed_hits=1, nf_false=0) == "anecdotal"
    assert CMP.classify_feature(failed_hits=4, nf_false=2) == "false_alarm_prone"
    assert CMP.classify_feature(failed_hits=0, nf_false=0) == "no_signal"
