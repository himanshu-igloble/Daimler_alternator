import importlib.util
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load("V11_ALT_heuristics_verify")


def test_scan_forbidden_clean_on_features():
    src = (_SRC / "V11_ALT_heuristics_features.py").read_text(encoding="utf-8")
    assert V.scan_forbidden(src) == []


def test_scan_forbidden_flags_sklearn():
    assert "sklearn" in str(V.scan_forbidden("import sklearn\nfrom x import Ridge("))
