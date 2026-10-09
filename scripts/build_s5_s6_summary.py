"""Build the normative S5 three-chain ablation and S6 efficiency summaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATASETS=("MOSI","MOSEI","CREMA-D","AV-MNIST")


def main():
    audit=pd.read_csv(ROOT/"results/i1_1_supervision_audit.csv")
    mechanism=pd.read_csv(ROOT/"results/e8_mechanism_chain.csv")
    single=pd.read_csv(ROOT/"results/e8_single_member_summary.csv")
    p0=pd.read_csv(ROOT/"results/p0_final_system_registry.csv")
    rows=[]
    for dataset in DATASETS:
        key=dataset.lower().replace("-","")
        ar=audit[audit.dataset==key].iloc[0]
        m=mechanism[mechanism.dataset==dataset]
        a1=m[m.stage=="A1"].iloc[0];a2=m[m.stage=="A2"].iloc[0]
        a3=m[m.stage=="A3"].iloc[0];a5=m[m.stage=="A5"].iloc[0]
        stages=single[single.dataset==dataset].set_index("stage")
        registry=p0[p0.dataset==key].iloc[0]
        rows.extend([
            {"dataset":dataset,"chain":"supervision","stage":"A1 OOF hard","primary_metric":"candidate_recovery","value":a1.candidate_recovery,"gain":0.},
            {"dataset":dataset,"chain":"supervision","stage":"A2 OOF soft","primary_metric":"candidate_recovery","value":a2.candidate_recovery,"gain":a2.candidate_recovery-a1.candidate_recovery},
            {"dataset":dataset,"chain":"supervision","stage":"soft variance ratio","primary_metric":"soft/hard target variance","value":ar.soft_to_hard_variance_ratio,"gain":1-ar.soft_to_hard_variance_ratio},
            {"dataset":dataset,"chain":"routing","stage":"A3 analytic","primary_metric":"candidate_recovery","value":a3.candidate_recovery,"gain":0.},
            {"dataset":dataset,"chain":"routing","stage":"A5 anchored listwise","primary_metric":"candidate_recovery","value":a5.candidate_recovery,"gain":a5.candidate_recovery-a3.candidate_recovery},
            {"dataset":dataset,"chain":"decision","stage":"A6 mixer","primary_metric":"single-member NLL gain","value":stages.loc["A6","nll_mean"],"gain":stages.loc["A6","nll_gain_mean"]},
            {"dataset":dataset,"chain":"decision","stage":"A7 risk control","primary_metric":"single-member NLL gain","value":stages.loc["A7","nll_mean"],"gain":stages.loc["A7","nll_gain_mean"]},
            {"dataset":dataset,"chain":"decision","stage":"A8 P0 v2","primary_metric":"ensemble NLL gain","value":registry.final_nll,"gain":registry.nll_gain},
        ])
    result=pd.DataFrame(rows);result.to_csv(ROOT/"results/s5_three_chain_ablation.csv",index=False)

    fig,axes=plt.subplots(1,3,figsize=(10.6,3.2))
    x=np.arange(4);colors=["#0072B2","#E69F00","#009E73","#CC79A7"]
    variance=[1-audit[audit.dataset==d.lower().replace('-','')].iloc[0].soft_to_hard_variance_ratio for d in DATASETS]
    route=[];a7=[];a8=[]
    for d in DATASETS:
        m=mechanism[mechanism.dataset==d].set_index("stage")
        route.append(m.loc["A5","candidate_recovery"]-m.loc["A3","candidate_recovery"])
        s=single[single.dataset==d].set_index("stage");a7.append(s.loc["A7","nll_gain_mean"])
        a8.append(p0[p0.dataset==d.lower().replace('-','')].iloc[0].nll_gain)
    axes[0].bar(x,variance,color=colors);axes[0].set_ylabel("Variance reduction fraction");axes[0].set_title("Supervision: soft vs hard")
    axes[1].bar(x,route,color=colors);axes[1].axhline(0,color='#555',lw=.8);axes[1].set_ylabel("Top-K recovery gain");axes[1].set_title("Routing: A5 vs A3")
    width=.36;axes[2].bar(x-width/2,a7,width,label='A7 single',color='#56B4E9');axes[2].bar(x+width/2,a8,width,label='A8 ensemble',color='#D55E00');axes[2].axhline(0,color='#555',lw=.8);axes[2].set_ylabel("NLL improvement");axes[2].set_title("Decision chain");axes[2].legend(frameon=False)
    for ax in axes:ax.set_xticks(x,DATASETS,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/"figures/s5_three_chain_ablation.png",dpi=300,facecolor='white');plt.close(fig)

    efficiency=pd.read_csv(ROOT/"results/e12_efficiency.csv")
    mosi=efficiency[(efficiency.dataset=='mosi')&(efficiency.batch_size==64)].set_index('configuration')
    full=mosi.loc['five-member full ensemble'];rcg=mosi.loc['five-member complete RCG']
    registry=p0[p0.dataset=='mosi'].iloc[0]
    incremental=rcg.latency_ms_mean-full.latency_ms_mean
    s6=pd.DataFrame([{
        "dataset":"mosi","deployment":"five-member P0 v2","batch_size":64,
        "full_latency_ms":full.latency_ms_mean,"rcg_latency_ms":rcg.latency_ms_mean,
        "incremental_latency_ms":incremental,"nll_improvement":registry.nll_gain,
        "incremental_ms_per_0.01_nll":incremental/(registry.nll_gain/.01),
        "latency_multiplier":rcg.latency_ms_mean/full.latency_ms_mean,
        "peak_vram_mb":rcg.peak_vram_mb,
    }])
    s6.to_csv(ROOT/"results/s6_benefit_normalized_efficiency.csv",index=False)


if __name__=="__main__":main()
