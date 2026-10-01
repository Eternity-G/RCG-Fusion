"""Build the preregistered observation figures and Markdown tables.

All manuscript figures are PNG-only, 300 dpi, white background, and use the
Okabe-Ito colorblind-safe palette.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


COLORS={"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
        "purple":"#CC79A7","sky":"#56B4E9","yellow":"#F0E442","black":"#222222"}


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
                         "legend.fontsize":7,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})


def save(fig,path):
    fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)


def records(root):
    return pd.concat([pd.read_parquet(p) for p in sorted(Path(root).glob("*/fold_*/*/seed_*/samples.parquet"))],ignore_index=True)


def figure1(t1,t2,seeds,rec,out,legacy_root):
    fig=plt.figure(figsize=(7.2,5.8)); gs=fig.add_gridspec(2,2,wspace=.34,hspace=.38)
    ax=fig.add_subplot(gs[0,0]); ax.axis("off"); ax.set_title("a  Concept",loc="left",fontweight="bold")
    boxes=[(.05,.58,.36,.22,"Modality m","reliability rₘ\nfrom xₘ alone",COLORS["blue"]),
           (.59,.58,.36,.22,"Current coalition S","other available\nmodalities",COLORS["green"]),
           (.30,.12,.42,.25,"Conditional contribution","uₘ(S)=L(S)−L(S∪{m})\ncan be positive or negative",COLORS["red"])]
    for x,y,w,h,title,body,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.02",fc="white",ec=color,lw=1.5))
        ax.text(x+w/2,y+h*.67,title,ha="center",va="center",weight="bold",color=color,fontsize=7)
        ax.text(x+w/2,y+h*.28,body,ha="center",va="center",fontsize=7)
    ax.add_patch(FancyArrowPatch((.23,.55),(.42,.38),arrowstyle="->",mutation_scale=10,color="#666"))
    ax.add_patch(FancyArrowPatch((.77,.55),(.60,.38),arrowstyle="->",mutation_scale=10,color="#666"))

    ax=fig.add_subplot(gs[0,1]); ax.set_title("b  High reliability can be harmful",loc="left",fontweight="bold")
    forest=t1[(t1.method=="concat")].copy(); forest=forest[forest.modality.isin(["audio","vision","visual"])]
    av_path=Path(legacy_root)/"avmnist-hf-pilot"/"metrics.csv"
    if av_path.exists():
        av=pd.read_csv(av_path); av=av[av.metric=="hcr_shapley"]
        rows=[]
        for modality,g in av.groupby("modality"):
            mean=g.estimate.mean(); half=2.776*g.estimate.std(ddof=1)/np.sqrt(len(g))
            rows.append({"dataset":"avmnist","method":"concat","modality":modality,"hcr":mean,
                         "hcr_ci_low":max(0,mean-half),"hcr_ci_high":min(1,mean+half)})
        forest=pd.concat([forest,pd.DataFrame(rows)],ignore_index=True)
    forest["label"]=forest.dataset.str.upper()+" · "+forest.modality
    forest=forest.sort_values(["dataset","modality"]); y=np.arange(len(forest))
    ax.errorbar(forest.hcr,y,xerr=[forest.hcr-forest.hcr_ci_low,forest.hcr_ci_high-forest.hcr],fmt="o",color=COLORS["red"],capsize=2)
    ax.axvline(.05,color="#777",ls="--",lw=1,label="5% continuation threshold"); ax.set_yticks(y,forest.label); ax.set_xlabel("HCR (Shapley < −0.01)"); ax.set_xlim(left=0); ax.legend(frameon=False)

    ax=fig.add_subplot(gs[1,0]); ax.set_title("c  Harm identification",loc="left",fontweight="bold")
    weak=seeds[seeds.modality.isin(["audio","vision","visual"])]
    plot=weak.groupby(["method","seed"],as_index=False).auroc.mean().groupby("method").auroc.agg(["mean","std"]).reset_index()
    x=np.arange(len(plot)); ax.bar(x,plot["mean"],yerr=plot["std"],color=[COLORS[k] for k in ["blue","orange","green","purple","red"][:len(plot)]],capsize=2)
    ax.axhline(.5,color="#555",ls="--",lw=1); ax.set_xticks(x,plot.method.str.upper(),rotation=25,ha="right"); ax.set_ylabel("AUROC for harmful addition"); ax.set_ylim(0,1)

    ax=fig.add_subplot(gs[1,1]); ax.set_title("d  Avoidable full-fusion loss",loc="left",fontweight="bold")
    plotted=[]
    for dataset,color in zip(sorted(rec.dataset.unique()),[COLORS["blue"],COLORS["orange"],COLORS["green"]]):
        values=np.sort(rec[(rec.dataset==dataset)&(rec.method=="concat")].fusion_regret.to_numpy()); y=np.arange(1,len(values)+1)/len(values)
        plotted.append(values)
        ax.plot(values,y,label=dataset.upper(),color=color,lw=1.5)
    av_records=sorted((Path(legacy_root)/"avmnist-hf-pilot").glob("seed_*.parquet"))
    if av_records:
        values=np.sort(pd.concat([pd.read_parquet(p) for p in av_records]).fusion_regret.to_numpy()); y=np.arange(1,len(values)+1)/len(values)
        plotted.append(values)
        ax.plot(values,y,label="AV-MNIST",color=COLORS["purple"],lw=1.5)
    ax.axvline(.01,color="#777",ls="--",lw=1); ax.axvline(.05,color="#aaa",ls=":",lw=1)
    upper=np.quantile(np.concatenate(plotted),.99) if plotted else 1
    ax.set_xlabel("Fusion regret (nats; x-axis to 99th pct.)"); ax.set_ylabel("ECDF"); ax.set_xlim(0,max(.05,upper)); ax.legend(frameon=False)
    save(fig,out/"figure1_intro_motivation.png")


def figure2(t1,out):
    data=t1[t1.modality.isin(["audio","vision","visual"])].copy(); methods=sorted(data.method.unique()); datasets=sorted(data.dataset.unique())
    fig,axes=plt.subplots(1,len(datasets),figsize=(3.0*len(datasets),3.0),sharey=True,squeeze=False)
    for ax,dataset in zip(axes[0],datasets):
        part=data[data.dataset==dataset]; labels=sorted(part.modality.unique()); width=.8/max(1,len(methods)); x=np.arange(len(labels))
        for j,method in enumerate(methods):
            row=part[part.method==method].set_index("modality").reindex(labels); pos=x+(j-(len(methods)-1)/2)*width
            ax.bar(pos,row.hcr,width,label=method.upper(),color=list(COLORS.values())[j]); ax.errorbar(pos,row.hcr,
                yerr=[row.hcr-row.hcr_ci_low,row.hcr_ci_high-row.hcr],fmt="none",ecolor="#333",capsize=1,lw=.7)
        ax.axhline(.05,color="#555",ls="--",lw=1); ax.set_title(dataset.upper()); ax.set_xticks(x,labels); ax.set_ylim(0,1)
    axes[0,0].set_ylabel("High-reliability negative-Shapley rate"); axes[0,-1].legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
    fig.suptitle("Figure 2 | Reliability–contribution gap across fusion mechanisms",fontweight="bold",y=1.03)
    save(fig,out/"figure2_cross_method_hcr.png")


def figure3(t1,out):
    data=t1.copy(); data["label"]=data.dataset.str.upper()+" · "+data.modality
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
    for ax,metric,title in zip(axes,["harm_auroc","harm_auprc"],["AUROC","AUPRC"]):
        pivot=data.pivot_table(index="label",columns="method",values=metric); y=np.arange(len(pivot)); offsets=np.linspace(-.28,.28,len(pivot.columns))
        for j,method in enumerate(pivot.columns): ax.scatter(pivot[method],y+offsets[j],s=20,label=method.upper(),color=list(COLORS.values())[j])
        if metric=="harm_auroc": ax.axvline(.5,color="#555",ls="--",lw=1)
        ax.set_yticks(y,pivot.index); ax.set_xlabel(title); ax.set_xlim(0,1)
    axes[1].legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
    fig.suptitle("Figure 3 | Can native reliability, quality, or routing scores detect harmful addition?",fontweight="bold")
    save(fig,out/"figure3_harm_detection.png")


def figure4(t2,out):
    methods=sorted(t2.method.unique()); datasets=sorted(t2.dataset.unique()); fig,axes=plt.subplots(1,3,figsize=(7.2,3.15))
    for i,(metric,title) in enumerate([("regret_gt_001","Regret > 0.01"),("negative_flip_any","Any negative flip")]):
        pivot=t2.pivot(index="dataset",columns="method",values=metric).reindex(datasets); pivot.plot.bar(ax=axes[i],color=list(COLORS.values())[:len(methods)],legend=False)
        axes[i].set_title(title); axes[i].set_xlabel(""); axes[i].set_ylim(bottom=0); axes[i].tick_params(axis="x",rotation=25)
    subset=t2.groupby("dataset")[["oracle_single_rate","oracle_pair_rate","oracle_full_rate"]].mean().reindex(datasets)
    subset.plot.bar(stacked=True,ax=axes[2],color=[COLORS["orange"],COLORS["sky"],COLORS["green"]]); axes[2].set_title("Oracle coalition composition"); axes[2].set_xlabel(""); axes[2].tick_params(axis="x",rotation=25); axes[2].legend(["Single","Pair","Full"],frameon=False,fontsize=6)
    fig.suptitle("Figure 4 | Actual decision harm from forced full fusion",fontweight="bold")
    handles=[Patch(facecolor=list(COLORS.values())[i],label=m.upper()) for i,m in enumerate(methods)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.87),ncol=len(methods),frameon=False)
    fig.subplots_adjust(top=.67,wspace=.25)
    save(fig,out/"figure4_prediction_harm.png")


def figure5(t4,out,legacy_root):
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8))
    part=t4.groupby(["epsilon","method"])[["hcr_deletion","hcr_shapley"]].mean().reset_index()
    for method,color in zip(sorted(part.method.unique()),list(COLORS.values())):
        x=part[part.method==method]; axes[0].plot(x.epsilon,x.hcr_shapley,"o-",color=color,label=method.upper(),ms=3)
    axes[0].set_title("Contribution tolerance"); axes[0].set_xlabel("ε (nats)"); axes[0].set_ylabel("HCR")
    compare=t4.groupby("method")[["hcr_deletion","hcr_shapley"]].mean(); compare.plot.bar(ax=axes[1],color=[COLORS["sky"],COLORS["red"]]); axes[1].set_title("Deletion vs Shapley"); axes[1].set_xlabel(""); axes[1].tick_params(axis="x",rotation=25); axes[1].legend(frameon=False,fontsize=6)
    stress=[]
    for dataset in ("mosi","mosei"):
        p=Path(legacy_root)/dataset/"analysis"/"diagnostics_summary.csv"
        if p.exists():
            d=pd.read_csv(p); d=d[(d.method=="concat_masked")&(d.score=="calibrated_probe")&(d.epsilon==.01)&(d.modality.isin(["audio","vision"]))]
            d=d[d.family=="main"]; d["dataset"]=dataset; stress.append(d)
    if stress:
        d=pd.concat(stress); labels={"clean":"Clean","gaussian":"Gaussian","mask":"Masking"}
        for kind,color in zip(["clean","gaussian","mask"],[COLORS["green"],COLORS["orange"],COLORS["purple"]]):
            q=d[d.kind==kind].groupby("severity").hcr_mean.mean(); axes[2].plot(q.index,q.values,"o-",label=labels[kind],color=color,ms=3)
    axes[2].set_title("Feature-space stress tests"); axes[2].set_xlabel("Severity"); axes[2].set_ylabel(""); axes[2].legend(frameon=False,fontsize=6)
    axes[0].legend(frameon=False,fontsize=6); fig.suptitle("Figure 5 | Protocol and stress-test robustness",fontweight="bold")
    fig.subplots_adjust(top=.76,wspace=.30)
    save(fig,out/"figure5_robustness_sensitivity.png")


def markdown_tables(analysis,out):
    labels={"table1_reliability_contribution.csv":"Table 1. Reliability–contribution diagnostics",
            "table2_prediction_harm.csv":"Table 2. Fusion regret and prediction harm",
            "table3_task_metrics.csv":"Table 3. Main task metrics",
            "table4_protocol_sensitivity.csv":"Table 4. Contribution-definition and tolerance sensitivity"}
    lines=[]
    for filename,title in labels.items():
        frame=pd.read_csv(analysis/filename).round(4); headers=[str(c) for c in frame.columns]
        table=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
        table.extend("| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
        lines += [f"## {title}","","\n".join(table),""]
    (out/"TABLES.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--strong-root",default="runs/strong-observation"); p.add_argument("--analysis",default="runs/observation-final/analysis")
    p.add_argument("--output",default="runs/observation-final"); p.add_argument("--legacy-root",default="runs"); a=p.parse_args()
    style(); analysis=Path(a.analysis); out=Path(a.output); figs=out/"figures"; source=out/"source_data"; figs.mkdir(parents=True,exist_ok=True); source.mkdir(parents=True,exist_ok=True)
    t1=pd.read_csv(analysis/"table1_reliability_contribution.csv"); t2=pd.read_csv(analysis/"table2_prediction_harm.csv"); t4=pd.read_csv(analysis/"table4_protocol_sensitivity.csv"); seeds=pd.read_csv(analysis/"seed_metrics.csv"); rec=records(a.strong_root)
    figure1(t1,t2,seeds,rec,figs,a.legacy_root); figure2(t1,figs); figure3(t1,figs); figure4(t2,figs); figure5(t4,figs,a.legacy_root)
    for pth in analysis.glob("*.csv"): shutil.copy2(pth,source/pth.name)
    markdown_tables(analysis,out)


if __name__=="__main__": main()
