import importlib.util, pathlib
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parents[1]/"src"
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=_load("proxy_anr_torque")

def test_idle_rows_keeps_only_engine_on_stationary_noncrank():
    df=pl.DataFrame({"RPM":[600.0,600.0,2000.0,600.0,600.0],"CSP":[0.0,1.0,0.0,50.0,0.0],
                     "SMA":[0,0,0,0,1],"ANR":[30.0,32.0,200.0,40.0,31.0]})
    assert P.idle_rows(df).height==2   # rows 0,1 only (row2 high-RPM, row3 moving, row4 cranking)

def test_idle_residual_zero_when_matches_baseline():
    di=pl.DataFrame({"RPM":[600.0,600.0],"CSP":[0.0,0.0],"SMA":[0,0],"ANR":[30.0,34.0]})
    base=P.fit_idle_baseline(di)            # median ANR in rpm-bin 12 = 32
    r=P.idle_residual(di,base)
    assert abs(float(r["anr_resid"].mean()))<1e-9   # residuals -2,+2 -> mean 0
