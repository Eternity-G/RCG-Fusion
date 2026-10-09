"""Analyze availability-safe RCG against coalition-valid missing baselines."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
BASELINES = ("coalition_dropout", "tmc", "qmf", "pdf", "i2moe")


def holm(values):
    values=np.asarray(values,float); order=np.argsort(values); out=np.empty_like(values); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(values)-rank)*values[index]);out[index]=min(running,1.)
    return out


def load(dataset):
    internal=pd.read_parquet(ROOT/f"runs/formal-e10-{dataset}/predictions.parquet")
    internal=internal[internal.method.isin(("coalition_dropout","availability_safe_rcg"))]
    strong=pd.read_parquet(ROOT/f"runs/formal-s3-strong-missing/{dataset}/predictions.parquet")
    return pd.concat([internal,strong],ignore_index=True)


def main():
    summaries=[]; boot_rows=[]
    for dataset in DATASETS:
        frame=load(dataset)
        full_mask=frame.available_mask.astype(str).str.count("1").max()
        frame=frame[frame.available_mask.astype(str).str.count("1")<full_mask].copy()
        pcols=sorted((c for c in frame if c.startswith('p') and c[1:].isdigit()),key=lambda x:int(x[1:]))
        probability=frame[pcols].to_numpy(float); labels=frame.label.to_numpy(int); row=np.arange(len(frame))
        frame["loss"]=-np.log(np.clip(probability[row,labels],1e-12,1));frame["correct"]=(probability.argmax(1)==labels).astype(float)
        for method,part in frame.groupby("method"):
            summaries.append({"dataset":dataset,"method":method,"accuracy":part.correct.mean(),"nll":part.loss.mean()})
        key=["fold","available_mask","sample_id"]
        loss=frame.pivot(index=key,columns="method",values="loss").dropna()
        acc=frame.pivot(index=key,columns="method",values="correct").dropna()
        meta=frame.drop_duplicates(key).set_index(key).group_or_video_id.astype(str)
        for baseline in BASELINES:
            common=loss[["availability_safe_rcg",baseline]].dropna().index
            difference=pd.DataFrame({
                "nll_gain":loss.loc[common,baseline]-loss.loc[common,"availability_safe_rcg"],
                "accuracy_gain":acc.loc[common,"availability_safe_rcg"]-acc.loc[common,baseline],
                "cluster":[f"{fold}:{meta.loc[(fold,mask,sample)]}" for fold,mask,sample in common],
            }).groupby("cluster")[["nll_gain","accuracy_gain"]].mean()
            rng=np.random.default_rng(20261009); values=difference.to_numpy(); draws=values[
                rng.integers(0,len(values),size=(10_000,len(values)))].mean(1)
            observed=values.mean(0)
            boot_rows.append({"dataset":dataset,"baseline":baseline,"n_clusters":len(values),
                "nll_gain":observed[0],"nll_ci_low":np.quantile(draws[:,0],.025),"nll_ci_high":np.quantile(draws[:,0],.975),
                "nll_p":min(1.,2*min((draws[:,0]<=0).mean(),(draws[:,0]>=0).mean())),
                "accuracy_gain":observed[1],"accuracy_ci_low":np.quantile(draws[:,1],.025),
                "accuracy_ci_high":np.quantile(draws[:,1],.975),
                "accuracy_p":min(1.,2*min((draws[:,1]<=0).mean(),(draws[:,1]>=0).mean()))})
    summary=pd.DataFrame(summaries);bootstrap=pd.DataFrame(boot_rows)
    for _,index in bootstrap.groupby("dataset").groups.items():
        bootstrap.loc[index,"nll_p_holm"]=holm(bootstrap.loc[index,"nll_p"])
        bootstrap.loc[index,"accuracy_p_holm"]=holm(bootstrap.loc[index,"accuracy_p"])
    summary.to_csv(ROOT/"results/s3_missing_summary.csv",index=False)
    bootstrap.to_csv(ROOT/"results/s3_missing_bootstrap.csv",index=False)

    methods=("coalition_dropout","tmc","qmf","pdf","i2moe","availability_safe_rcg")
    labels_map={"coalition_dropout":"Coalition","availability_safe_rcg":"RCG","tmc":"TMC","qmf":"QMF","pdf":"PDF","i2moe":"I²MoE"}
    colors=["#999999","#009E73","#CC79A7","#E69F00","#56B4E9","#0072B2"]
    fig,axes=plt.subplots(1,4,figsize=(12.2,3.2))
    for ax,dataset in zip(axes,DATASETS):
        part=summary[summary.dataset==dataset].set_index("method").loc[list(methods)]
        ax.bar(range(len(methods)),part.nll,color=colors);ax.set_title(dataset.upper());ax.grid(axis="y",alpha=.2)
        ax.set_xticks(range(len(methods)),[labels_map[m] for m in methods],rotation=45,ha="right")
    axes[0].set_ylabel("NLL under incomplete availability")
    fig.tight_layout();fig.savefig(ROOT/"figures/s3_missing_modalities.png",dpi=300,facecolor="white");plt.close(fig)


if __name__=="__main__":main()
