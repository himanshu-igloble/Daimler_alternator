import importlib.util, pathlib
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=_load("analysis_2d_triggers")

def test_balanced_sample_equal_classes():
    df = pl.DataFrame({"GED":[2]*10 + [0]*1000, "VSI":[25.0]*1010})
    bs = T.balanced_sample(df, cap=100, seed=0)
    n2 = (bs["GED"]==2).sum(); n0 = (bs["GED"]==0).sum()
    assert n2 == 10 and n0 == 10        # matched to the minority count
