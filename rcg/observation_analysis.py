"""Aggregate strong-baseline sample records into the four preregistered tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def cluster_mean_ci(values, groups, draws=10_000, seed=90210):
    values=np.asarray(values,float); groups=np.asarray(groups); unique=np.unique(groups)
    sums=np.array([values[groups==g].sum() for g in unique]); counts=np.array([(groups==g).sum() for g in unique])
    rng=np.random.default_rng(seed); out=np.empty(draws)
    for start in range(0,draws,500):
        size=min(500,draws-start); idx=rng.integers(0,len(unique),(size,len(unique)))
        out[start:start+size]=sums[idx].sum(1)/counts[idx].sum(1)
    return float(values.mean()), *np.quantile(out,[.025,.975]).tolist()


def cluster_ratio_ci(numerator, denominator, groups, draws=10_000, seed=90210):
    numerator=np.asarray(numerator,float); denominator=np.asarray(denominator,float); groups=np.asarray(groups)
    unique=np.unique(groups); nums=np.array([numerator[groups==g].sum() for g in unique]); dens=np.array([denominator[groups==g].sum() for g in unique])
    estimate=numerator.sum()/denominator.sum() if denominator.sum() else np.nan
    rng=np.random.default_rng(seed); out=[]
    for start in range(0,draws,500):
        size=min(500,draws-start); idx=rng.integers(0,len(unique),(size,len(unique)))
        d=dens[idx].sum(1); n=nums[idx].sum(1); out.extend((n[d>0]/d[d>0]).tolist())
    low,high=np.quantile(out,[.025,.975]) if out else (np.nan,np.nan)
    return float(estimate),float(low),float(high)


def ece(probabilities, labels, bins=10):
    confidence=probabilities.max(1); prediction=probabilities.argmax(1); edges=np.linspace(0,1,bins+1); value=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        chosen=(confidence>=lo)&(confidence<(hi if hi<1 else hi+1e-12))
        if chosen.any(): value += chosen.mean()*abs((prediction[chosen]==labels[chosen]).mean()-confidence[chosen].mean())
    return float(value)


def probability_matrix(frame, prefix):
    cols=sorted([c for c in frame if c.startswith(prefix) and c[len(prefix):].isdigit()], key=lambda x:int(x[len(prefix):]))
    cols=[c for c in cols if np.isfinite(frame[c].to_numpy(float)).any()]
    return frame[cols].to_numpy(float)


def infer_modalities(frame):
    return [c.removeprefix("native_score_") for c in frame
            if c.startswith("native_score_") and np.isfinite(frame[c].to_numpy(float)).any()]


def load_records(root):
    paths=sorted(Path(root).glob("*/fold_*/*/seed_*/samples.parquet"))
    if not paths: raise FileNotFoundError(f"No strong observation records under {root}")
    return pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)


def analyze_root(root, output, draws=10_000):
    data=load_records(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    table1=[]; table2=[]; table3=[]; table4=[]; seed_rows=[]
    for (dataset,method), frame in data.groupby(["dataset","method"],sort=True):
        groups=frame.group_or_video_id.to_numpy(); labels=frame.label.to_numpy(); full=probability_matrix(frame,"full_p")
        oracle=probability_matrix(frame,"oracle_p"); modalities=infer_modalities(frame)
        for modality in modalities:
            score=frame[f"native_score_{modality}"].to_numpy(float); high=frame[f"high_reliability_{modality}"].to_numpy(bool)
            deletion=frame[f"deletion_contribution_{modality}"].to_numpy(float); shapley=frame[f"shapley_contribution_{modality}"].to_numpy(float)
            harmful=shapley<-.01; hcr=cluster_ratio_ci(high&harmful,high,groups,draws)
            valid=np.isfinite(score)&np.isfinite(shapley)
            if harmful[valid].min()!=harmful[valid].max():
                auroc=roc_auc_score(harmful[valid],-score[valid]); auprc=average_precision_score(harmful[valid],-score[valid])
            else: auroc=auprc=np.nan
            rho=spearmanr(score[valid],shapley[valid]).statistic
            harm=np.maximum(-shapley,0); avg_harm=cluster_ratio_ci(harm*high,high,groups,draws)
            table1.append({"dataset":dataset,"method":method,"modality":modality,"hcr":hcr[0],"hcr_ci_low":hcr[1],"hcr_ci_high":hcr[2],
                           "n_high":int(high.sum()),"harm_prevalence":harmful.mean(),"harm_auroc":auroc,"harm_auprc":auprc,
                           "spearman_reliability_contribution":rho,"high_reliability_mean_harm":avg_harm[0]})
            for epsilon in (0.,.01,.05):
                d_harm=deletion < -epsilon; s_harm=shapley < -epsilon
                table4.append({"dataset":dataset,"method":method,"modality":modality,"epsilon":epsilon,
                               "hcr_deletion":(d_harm&high).sum()/high.sum(),"hcr_shapley":(s_harm&high).sum()/high.sum(),
                               "sign_agreement":np.mean(np.sign(deletion)==np.sign(shapley)),
                               "rank_correlation":spearmanr(deletion,shapley).statistic})
            for seed, sf in frame.groupby("train_seed"):
                sh=sf[f"shapley_contribution_{modality}"].to_numpy() < -.01; hi=sf[f"high_reliability_{modality}"].to_numpy(bool)
                sc=sf[f"native_score_{modality}"].to_numpy(); valid_seed=np.isfinite(sc)
                seed_rows.append({"dataset":dataset,"method":method,"seed":seed,"modality":modality,
                                  "hcr":(sh&hi).sum()/hi.sum() if hi.sum() else np.nan,
                                  "auroc":roc_auc_score(sh[valid_seed],-sc[valid_seed]) if len(np.unique(sh[valid_seed]))==2 else np.nan,
                                  "auprc":average_precision_score(sh[valid_seed],-sc[valid_seed]) if sh[valid_seed].any() else np.nan})
        regret=frame.fusion_regret.to_numpy(float); regret_ci=cluster_mean_ci(regret,groups,draws)
        pred=full.argmax(1); opred=oracle.argmax(1)
        negative_cols=[c for c in frame if c.startswith("negative_flip_")]
        correction_cols=[c for c in frame if c.startswith("correction_by_deletion_")]
        oracle_size=frame.oracle_coalition.astype(str).str.count(r"\+")+1
        table2.append({"dataset":dataset,"method":method,"mean_regret":regret_ci[0],"regret_ci_low":regret_ci[1],"regret_ci_high":regret_ci[2],
                       "regret_gt_001":np.mean(regret>.01),"regret_gt_005":np.mean(regret>.05),
                       "negative_flip_any":frame[negative_cols].any(axis=1).mean(),"correction_by_deletion_any":frame[correction_cols].any(axis=1).mean(),
                       "full_accuracy":np.mean(pred==labels),"oracle_accuracy":np.mean(opred==labels),
                       "full_macro_f1":f1_score(labels,pred,average="macro"),"oracle_macro_f1":f1_score(labels,opred,average="macro"),
                       "full_nll":frame.full_loss.mean(),"oracle_nll":-np.log(np.maximum(oracle[np.arange(len(labels)),labels],1e-12)).mean(),
                       "oracle_single_rate":np.mean(oracle_size==1),"oracle_pair_rate":np.mean(oracle_size==2) if len(modalities)>2 else 0.,
                       "oracle_full_rate":np.mean(oracle_size==len(modalities))})
        onehot=np.eye(full.shape[1])[labels]
        table3.append({"dataset":dataset,"method":method,"accuracy":np.mean(pred==labels),"macro_f1":f1_score(labels,pred,average="macro"),
                       "nll":frame.full_loss.mean(),"brier":np.mean(np.square(full-onehot).sum(1)),"ece_10_equal_width":ece(full,labels)})
    outputs={"table1_reliability_contribution.csv":pd.DataFrame(table1),"table2_prediction_harm.csv":pd.DataFrame(table2),
             "table3_task_metrics.csv":pd.DataFrame(table3),"table4_protocol_sensitivity.csv":pd.DataFrame(table4),
             "seed_metrics.csv":pd.DataFrame(seed_rows)}
    for name,frame in outputs.items(): frame.to_csv(output/name,index=False)
    metadata={"cluster_bootstrap_draws":draws,"cluster":"group_or_video_id; all seeds and conditions retained within cluster",
              "harm_definition":"exact Shapley contribution < -0.01 nats","hcr_threshold":"fold/seed calibration-set native-score q90"}
    (output/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return outputs


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--output",required=True); p.add_argument("--draws",type=int,default=10_000)
    a=p.parse_args(); analyze_root(a.root,a.output,a.draws)


if __name__=="__main__": main()
