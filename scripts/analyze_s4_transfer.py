"""Apply canonical P0 v2 aggregation to S4 backbone-transfer adapters."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rcg.final_system import fit_final_system

ROOT=Path(__file__).resolve().parents[1]
DATASETS=("mosi","cremad"); BACKBONES=("concat","tmc","qmf","pdf","i2moe")
SEEDS=(11,22,33,44,55)


def probability(frame, variant, seed, pcols):
    return frame[(frame.variant==variant)&(frame.train_seed==seed)].sort_values("sample_id")[pcols].to_numpy(float)


def main():
    prediction_rows=[]; metric_rows=[]; bootstrap_rows=[]
    for dataset in DATASETS:
        root=ROOT/f"runs/formal-s4-backbone-transfer-{dataset}"
        test=pd.read_parquet(root/"predictions.parquet");selection=pd.read_parquet(root/"selection_predictions.parquet")
        pcols=sorted((c for c in test if c.startswith('p') and c[1:].isdigit()),key=lambda x:int(x[1:]))
        for backbone in BACKBONES:
            fold_frames=[]
            for fold in sorted(test.fold.unique()):
                te=test[(test.backbone==backbone)&(test.fold==fold)]
                se=selection[(selection.backbone==backbone)&(selection.fold==fold)]
                test_ids=np.sort(te.sample_id.unique());selection_ids=np.sort(se.sample_id.unique())
                full_sel=np.asarray([probability(se,"base",seed,pcols) for seed in SEEDS])
                a7_sel=np.asarray([probability(se,"complete_rcg",seed,pcols) for seed in SEEDS])
                full_test=np.asarray([probability(te,"base",seed,pcols) for seed in SEEDS])
                a7_test=np.asarray([probability(te,"complete_rcg",seed,pcols) for seed in SEEDS])
                label_sel=se[(se.train_seed==SEEDS[0])&(se.variant=="base")].sort_values("sample_id").label.to_numpy(int)
                metadata=te[(te.train_seed==SEEDS[0])&(te.variant=="base")].sort_values("sample_id")
                fitted=fit_final_system(full_sel,a7_sel,label_sel,full_test,a7_test,member_ids=SEEDS)
                for method,prob in (("base_ensemble",fitted.full_ensemble),("complete_equal",fitted.a7_ensemble),("p0_v2",fitted.final_probability)):
                    frame=pd.DataFrame({"dataset":dataset,"backbone":backbone,"fold":fold,"method":method,
                        "sample_id":test_ids,"group_or_video_id":metadata.group_or_video_id.astype(str).to_numpy(),
                        "label":metadata.label.to_numpy(int)})
                    for k in range(prob.shape[1]):frame[f"p{k}"]=prob[:,k]
                    fold_frames.append(frame)
            combined=pd.concat(fold_frames,ignore_index=True);prediction_rows.append(combined)
            for method,part in combined.groupby("method"):
                prob=part[pcols].to_numpy(float);labels=part.label.to_numpy(int);row=np.arange(len(part))
                metric_rows.append({"dataset":dataset,"backbone":backbone,"method":method,
                    "accuracy":float((prob.argmax(1)==labels).mean()),
                    "nll":float(-np.log(np.clip(prob[row,labels],1e-12,1)).mean())})
            base=combined[combined.method=="base_ensemble"].set_index(["fold","sample_id"])
            final=combined[combined.method=="p0_v2"].set_index(["fold","sample_id"]).loc[base.index]
            bp=base[pcols].to_numpy(float);fp=final[pcols].to_numpy(float);labels=base.label.to_numpy(int);row=np.arange(len(base))
            paired=pd.DataFrame({"cluster":base.group_or_video_id.astype(str).to_numpy(),
                "nll_gain":-np.log(np.clip(bp[row,labels],1e-12,1))+np.log(np.clip(fp[row,labels],1e-12,1)),
                "accuracy_gain":(fp.argmax(1)==labels).astype(float)-(bp.argmax(1)==labels).astype(float)}).groupby("cluster").mean()
            values=paired.to_numpy();rng=np.random.default_rng(20261009);boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            bootstrap_rows.append({"dataset":dataset,"backbone":backbone,"n_clusters":len(values),
                "nll_gain":values[:,0].mean(),"nll_ci_low":np.quantile(boot[:,0],.025),"nll_ci_high":np.quantile(boot[:,0],.975),
                "accuracy_gain":values[:,1].mean(),"accuracy_ci_low":np.quantile(boot[:,1],.025),"accuracy_ci_high":np.quantile(boot[:,1],.975)})
    predictions=pd.concat(prediction_rows,ignore_index=True);metrics=pd.DataFrame(metric_rows);bootstrap=pd.DataFrame(bootstrap_rows)
    predictions.to_parquet(ROOT/"results/s4_transfer_predictions.parquet",index=False)
    metrics.to_csv(ROOT/"results/s4_transfer_metrics.csv",index=False);bootstrap.to_csv(ROOT/"results/s4_transfer_bootstrap.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),sharey=True)
    for ax,dataset in zip(axes,DATASETS):
        part=bootstrap[bootstrap.dataset==dataset];x=np.arange(len(part));values=part.nll_gain.to_numpy()
        error=np.vstack((values-part.nll_ci_low.to_numpy(),part.nll_ci_high.to_numpy()-values))
        ax.bar(x,values,color="#0072B2",yerr=error,capsize=3);ax.axhline(0,color="#555",lw=.8)
        ax.set_xticks(x,[b.upper() if b!='i2moe' else 'I²MoE' for b in part.backbone]);ax.set_title(dataset.upper());ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel("P0 v2 NLL improvement over backbone ensemble")
    fig.tight_layout();fig.savefig(ROOT/"figures/s4_backbone_transfer.png",dpi=300,facecolor='white');plt.close(fig)


if __name__=="__main__":main()
