"""Phase 2D: empirical GED=2 trigger reverse-engineering."""
import importlib.util, pathlib, json
import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
FEATS=["VSI","RPM","ANR","CSP","SMA"]

def balanced_sample(df: pl.DataFrame, cap=20000, seed=0) -> pl.DataFrame:
    pos=df.filter(pl.col("GED")==2); neg=df.filter(pl.col("GED")==0)
    k=min(pos.height, neg.height, cap)
    if k==0: return df.head(0)
    return pl.concat([pos.sample(k, seed=seed), neg.sample(k, seed=seed)])

def main():
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    rows=[]
    for vin in C.FAILED_VINS:
        df=C.load_vin(vin).select(["GED"]+FEATS).drop_nulls()
        bs=balanced_sample(df, seed=0)
        if bs.height: rows.append(bs)
    data=pl.concat(rows)
    X=data.select(FEATS).to_numpy(); y=(data["GED"]==2).to_numpy().astype(int)
    Xs=StandardScaler().fit_transform(X)
    lr=LogisticRegression(max_iter=1000).fit(Xs,y)
    gb=GradientBoostingClassifier(random_state=0).fit(X,y)
    imp=permutation_importance(gb,X,y,n_repeats=5,random_state=0)
    out=pl.DataFrame({"feature":FEATS,"logit_coef":lr.coef_[0].tolist(),
                      "perm_importance":imp.importances_mean.tolist()}).sort("perm_importance",descending=True)
    out.write_csv(C.RESULTS/"2d_trigger_importance.csv")
    rep={"n_samples":int(data.height),"lr_intercept":float(lr.intercept_[0]),"top_trigger":out["feature"][0]}
    (C.RESULTS/"2d_trigger_report.json").write_text(json.dumps(rep,indent=2)); print(out); print(rep)

if __name__=="__main__":
    main()
