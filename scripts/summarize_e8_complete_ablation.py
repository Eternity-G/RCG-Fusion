"""Build the honest two-chain E8 ablation and its paired uncertainty analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"MOSI": "mosi", "MOSEI": "mosei", "CREMA-D": "cremad", "AV-MNIST": "avmnist"}
SINGLE = {"A0": "full_coalition", "A3": "analytic_hard", "A4": "listwise_hard",
          "A6": "anchored_harm_oracle", "A7": "analytic_pi_g2"}
COLORS = {"A0": "#777777", "A3": "#E69F00", "A4": "#56B4E9", "A6": "#009E73", "A7": "#0072B2",
          "A7_equal_ensemble": "#56B4E9", "A8_joint_convex": "#009E73", "A8_safe_fallback": "#0072B2"}


def ci95(values):
    values = np.asarray(values, float)
    return float(stats.t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan


def holm(values):
    values=np.asarray(values,float);order=np.argsort(values);result=np.empty(len(values));running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(values)-rank)*values[index]));result[index]=running
    return result


def fold_weighted(frame, keys):
    rows=[]; excluded=set(keys)|{"fold","train_seed","n_test","dataset","variant"}
    numeric=[c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
    for group, values in frame.groupby([*keys,"train_seed"]):
        group=group if isinstance(group,tuple) else (group,); w=values.n_test.to_numpy(float)
        row=dict(zip([*keys,"train_seed"],group));row["n_test"]=int(w.sum())
        row.update({m:float(np.average(values[m],weights=w)) for m in numeric});rows.append(row)
    return pd.DataFrame(rows)


def single_member_chain():
    e5=pd.read_csv(ROOT/"results/e5_candidate_mixer_by_seed.csv")
    e6=pd.read_csv(ROOT/"results/e6_shrinkage_by_seed.csv")
    rows=[]
    for dataset in DATASETS:
        for stage, variant in SINGLE.items():
            source=e6 if stage=="A7" else e5
            part=source[(source.dataset==dataset)&(source.variant==variant)].copy()
            for _,r in part.iterrows():
                rows.append({"dataset":dataset,"stage":stage,"variant":variant,"train_seed":int(r.train_seed),
                             **{m:float(r[m]) for m in ["accuracy","macro_f1","nll","brier","ece","accuracy_gain",
                                 "nll_gain","negative_flip_rate","correction_rate","clipped_harm"]}})
    return pd.DataFrame(rows)


def mechanism_chain():
    rows=[]
    for dataset,slug in DATASETS.items():
        e2=pd.read_csv(ROOT/f"results/e2_{slug}_supervision_ablation_by_seed.csv")
        for stage,variant in [("control_in_sample","in_sample_hard"),("A1","oof_multi_hard"),("A2","oof_soft_full")]:
            part=e2[e2.variant==variant]
            rows.append({"dataset":dataset,"stage":stage,"variant":variant,"n_seeds":len(part),
                         "ndcg":part.router_ndcg.mean(),"pairwise_accuracy":part.router_pairwise_accuracy.mean(),
                         "top2_recovery":part.router_top2.mean(),"candidate_recovery":part.anchored_top3.mean(),
                         "candidate_oracle_nll":part.candidate_oracle_nll.mean()})
    e4=pd.read_csv(ROOT/"results/e4_routing_by_seed.csv")
    for dataset in DATASETS:
        for stage,variant in [("A3","analytic_only"),("A4","listwise_pairwise_unanchored"),
                              ("A5","listwise_pairwise_anchored")]:
            part=e4[(e4.dataset==dataset)&(e4.variant==variant)]
            rows.append({"dataset":dataset,"stage":stage,"variant":variant,"n_seeds":len(part),
                         "ndcg":part.ndcg.mean(),"pairwise_accuracy":part.pairwise_accuracy.mean(),
                         "top2_recovery":part.top2_recovery.mean(),"candidate_recovery":part.candidate_oracle_recovery.mean(),
                         "candidate_oracle_nll":part.candidate_oracle_nll.mean()})
    return pd.DataFrame(rows)


def ensemble_bridge():
    values=[]; predictions=[]
    for dataset,slug in DATASETS.items():
        run=ROOT/f"runs/formal-e8-{slug}"
        frame=pd.read_csv(run/"metrics.csv");frame["dataset"]=dataset;values.append(frame)
        pred=pd.read_parquet(run/"predictions.parquet");pred["dataset"]=dataset;predictions.append(pred)
    return pd.concat(values,ignore_index=True),pd.concat(predictions,ignore_index=True)


def bootstrap(predictions,dataset,candidate,reference,repetitions=10_000,seed=8808):
    sub=predictions[(predictions.dataset==dataset)&predictions.method.isin([candidate,reference])]
    pcols=sorted([c for c in sub if c.startswith("p") and not sub[c].isna().all()],key=lambda x:int(x[1:]))
    keys=["fold","sample_id","group_or_video_id","label"]
    wide=sub.pivot(index=keys,columns="method",values=pcols); records=[]
    for index,row in wide.iterrows():
        fold,sample,cluster,label=index
        a=np.array([row[(c,candidate)] for c in pcols]);b=np.array([row[(c,reference)] for c in pcols])
        records.append({"cluster":cluster,"label":label,"acc":float(a.argmax()==label)-float(b.argmax()==label),
                        "nll":-np.log(np.clip(a[label],1e-12,1))+np.log(np.clip(b[label],1e-12,1))})
    v=pd.DataFrame(records);rng=np.random.default_rng(seed);acc=np.empty(repetitions);nll=np.empty(repetitions)
    if dataset=="AV-MNIST":
        indices=[np.flatnonzero(v.label.to_numpy()==k) for k in sorted(v.label.unique())]
        for i in range(repetitions):
            ix=np.concatenate([rng.choice(x,len(x),replace=True) for x in indices]);acc[i]=v.acc.to_numpy()[ix].mean();nll[i]=v.nll.to_numpy()[ix].mean()
        unit="class-stratified sample"
    else:
        cluster=v.cluster.astype(str);unique=np.unique(cluster)
        sums=v.assign(cluster=cluster).groupby("cluster")[["acc","nll"]].sum().reindex(unique)
        counts=v.assign(cluster=cluster).groupby("cluster").size().reindex(unique).to_numpy()
        for i in range(repetitions):
            ix=rng.integers(0,len(unique),len(unique));den=counts[ix].sum();acc[i]=sums.acc.to_numpy()[ix].sum()/den;nll[i]=sums.nll.to_numpy()[ix].sum()/den
        unit="actor" if dataset=="CREMA-D" else "original video"
    return {"dataset":dataset,"candidate":candidate,"reference":reference,"unit":unit,
            "accuracy_difference":v.acc.mean(),"accuracy_ci_low":np.quantile(acc,.025),"accuracy_ci_high":np.quantile(acc,.975),
            "nll_difference":v.nll.mean(),"nll_ci_low":np.quantile(nll,.025),"nll_ci_high":np.quantile(nll,.975)}


def plot(single,mechanism,bridge):
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7.3,
                         "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5),constrained_layout=True)
    for ax,metric,title in [(axes[0,0],"nll_gain","(a) Single-member NLL gain"),(axes[0,1],"accuracy_gain","(b) Single-member accuracy gain")]:
        stages=list(SINGLE);x=np.arange(len(DATASETS));width=.15
        for j,stage in enumerate(stages):
            means=[];errors=[]
            for dataset in DATASETS:
                part=single[(single.dataset==dataset)&(single.stage==stage)][metric]
                means.append(part.mean());errors.append(ci95(part))
            ax.bar(x+(j-2)*width,means,width,yerr=errors,capsize=1.5,label=stage,color=COLORS[stage])
        ax.axhline(0,color="#333",linewidth=.7);ax.set_xticks(x,list(DATASETS),rotation=15);ax.set_title(title,loc="left",fontweight="bold")
        ax.set_ylabel("Improvement vs member full coalition" if metric=="nll_gain" else "Accuracy change")
    axes[0,0].legend(ncol=3,fontsize=6.5)
    ax=axes[1,0];stages=["A3","A4","A5"];x=np.arange(len(DATASETS));width=.22
    for j,stage in enumerate(stages):
        vals=[mechanism[(mechanism.dataset==d)&(mechanism.stage==stage)].candidate_recovery.iloc[0] for d in DATASETS]
        ax.bar(x+(j-1)*width,vals,width,label=stage,color=COLORS.get(stage,"#999"))
    ax.set_ylim(0,1);ax.set_xticks(x,list(DATASETS),rotation=15);ax.set_ylabel("Candidate-oracle recovery");ax.set_title("(c) Candidate coverage",loc="left",fontweight="bold");ax.legend()
    ax=axes[1,1];methods=["A7_equal_ensemble","A8_joint_convex","A8_safe_fallback"];x=np.arange(len(DATASETS));width=.22
    for j,method in enumerate(methods):
        vals=[]
        for d in DATASETS:
            row=bridge[(bridge.dataset==d)&(bridge.method==method)].iloc[0];vals.append(row.nll_gain)
        ax.bar(x+(j-1)*width,vals,width,label=method.replace("_"," "),color=COLORS[method])
    ax.axhline(0,color="#333",linewidth=.7);ax.set_xticks(x,list(DATASETS),rotation=15);ax.set_ylabel("NLL gain vs full ensemble");ax.set_title("(d) A7-to-A8 bridge",loc="left",fontweight="bold");ax.legend(fontsize=6.2)
    fig.savefig(ROOT/"figures/e8_complete_ablation.png",dpi=300,facecolor="white",bbox_inches="tight");plt.close(fig)


def main():
    result=ROOT/"results";single=single_member_chain();mechanism=mechanism_chain();bridge,predictions=ensemble_bridge()
    single.to_csv(result/"e8_single_member_chain.csv",index=False);mechanism.to_csv(result/"e8_mechanism_chain.csv",index=False);bridge.to_csv(result/"e8_ensemble_bridge.csv",index=False)
    summaries=[]
    for (dataset,stage),v in single.groupby(["dataset","stage"]):
        row={"dataset":dataset,"stage":stage,"n_seeds":len(v)}
        for m in ["accuracy","macro_f1","nll","accuracy_gain","nll_gain","negative_flip_rate","correction_rate","clipped_harm"]:
            row[m+"_mean"]=v[m].mean();row[m+"_ci95"]=ci95(v[m])
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(result/"e8_single_member_summary.csv",index=False)
    tests=[]
    for dataset in DATASETS:
        part=single[single.dataset==dataset]
        for left,right in [("A3","A0"),("A4","A3"),("A6","A4"),("A7","A6")]:
            for metric in ["accuracy","nll","clipped_harm"]:
                a=part[part.stage==left].sort_values("train_seed")[metric].to_numpy()
                b=part[part.stage==right].sort_values("train_seed")[metric].to_numpy();d=a-b;t=stats.ttest_rel(a,b)
                tests.append({"dataset":dataset,"comparison":f"{left}_vs_{right}","metric":metric,
                              "mean_difference":d.mean(),"ci95":ci95(d),"p_value":t.pvalue})
    adjusted=holm([x["p_value"] for x in tests])
    for row,value in zip(tests,adjusted):row["holm_p"]=value
    pd.DataFrame(tests).to_csv(result/"e8_single_member_paired_tests.csv",index=False)
    boots=[]
    for dataset in DATASETS:
        for reference in ["A0_full_ensemble","A7_equal_ensemble"]:
            boots.append(bootstrap(predictions,dataset,"A8_safe_fallback",reference))
    pd.DataFrame(boots).to_csv(result/"e8_bridge_bootstrap.csv",index=False)
    plot(single,mechanism,bridge)
    print(bridge[["dataset","method","accuracy","nll","accuracy_gain","nll_gain","negative_flip_rate","correction_rate","clipped_harm"]].to_string(index=False))


if __name__=="__main__":main()
