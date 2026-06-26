import importlib.util, pathlib, datetime as dt
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=_load("analysis_2c_markov")

def test_transition_counts_respects_gap_and_nulls():
    base = dt.datetime(2025,1,1,0,0,0)
    ts = [base, base+dt.timedelta(seconds=5), base+dt.timedelta(seconds=10),
          base+dt.timedelta(seconds=10000), base+dt.timedelta(seconds=10005)]
    df = pl.DataFrame({"timestamp":ts, "GED":[0,2,None,0,0]})
    cm = M.transition_counts(df, dt_col="timestamp", max_gap_s=60)
    # 0->2 counted; 2->null(=3) counted; null gap (10000s) NOT counted; 0->0 counted
    assert cm[(0,2)] == 1 and cm[(2,3)] == 1 and cm[(0,0)] == 1
    assert sum(cm.values()) == 3   # the big jump is excluded
