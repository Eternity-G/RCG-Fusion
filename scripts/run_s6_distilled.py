"""Distill the frozen MOSI P0-v2 teacher into one coalition backbone."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from rcg.rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset, seed_all

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_e4_anchored_routing import SEEDS  # noqa: E402
from run_e9_clean_stress import load_context, member_predict  # noqa: E402


def teacher(split, contexts, masks, weights, rho, device):
    values=[member_predict(context,split,masks,3,device) for context in contexts]
    full=np.mean([value['full_ensemble'] for value in values],axis=0)
    a7=np.asarray([value['A7_equal_ensemble'] for value in values])
    convex=np.tensordot(weights,a7,axes=(0,0))
    return ((1-rho)*full+rho*convex).astype(np.float32)


@torch.inference_mode()
def predict(model,split,device,batch=1024):
    result=[]
    for start in range(0,len(split['y']),batch):
        stop=min(len(split['y']),start+batch);xs=[torch.as_tensor(x[start:stop],dtype=torch.float32,device=device) for x in split['x']]
        mask=torch.ones((stop-start,len(xs)),device=device);result.append(torch.softmax(model(xs,mask)['logits'],-1).cpu().numpy())
    return np.concatenate(result)


def train_student(seed,splits,teachers,device):
    dims=[x.shape[1] for x in splits['train']['x']];model=CoalitionAwareBackbone(dims,2).to(device);seed_all(seed)
    xs=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in splits['train']['x']];y=torch.as_tensor(splits['train']['y'],dtype=torch.long,device=device)
    teacher_t=torch.as_tensor(teachers['train'],dtype=torch.float32,device=device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    best,state,stale=float('inf'),None,0
    for _ in range(100):
        model.train()
        for index in torch.randperm(len(y),device=device).split(64):
            bx=[x[index] for x in xs];mask=torch.ones((len(index),len(xs)),device=device);logits=model(bx,mask)['logits']
            hard=F.cross_entropy(logits,y[index]);soft=-(teacher_t[index]*F.log_softmax(logits,-1)).sum(1).mean();loss=.5*hard+.5*soft
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval();prob=predict(model,splits['selection'],device);labels=splits['selection']['y'];nll=float(-np.log(np.clip(prob[np.arange(len(labels)),labels],1e-12,1)).mean())
        if nll<best-1e-6:best,state,stale=nll,copy.deepcopy(model.state_dict()),0
        else:stale+=1
        if stale>=10:break
    model.load_state_dict(state);model.eval();return model,best


def benchmark(model,split,device,batch_size,repeats):
    index=np.arange(batch_size)%len(split['y']);xs=[torch.as_tensor(x[index],dtype=torch.float32,device=device) for x in split['x']];mask=torch.ones((batch_size,len(xs)),device=device)
    fn=lambda:torch.softmax(model(xs,mask)['logits'],-1)
    for _ in range(20):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(7):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(repeats):fn()
        torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000/repeats)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();fn();torch.cuda.synchronize()
    return np.mean(samples),np.std(samples,ddof=1),torch.cuda.max_memory_allocated()/1024**2


def main():
    device='cuda' if torch.cuda.is_available() else 'cpu';folds,names=load_dataset('mosi',ROOT/'data');splits=folds[0];masks=nonempty_coalitions(len(names))
    contexts=[load_context('mosi',0,seed,masks,device) for seed in SEEDS]
    weights=pd.read_csv(ROOT/'runs/formal-e8-mosi/weights.csv').weight.to_numpy(float)
    rho=float(pd.read_csv(ROOT/'runs/formal-e8-mosi/metrics_by_fold.csv').query("method == 'A8_safe_fallback'").rho.iloc[0])
    teachers={name:teacher(splits[name],contexts,masks,weights,rho,device) for name in ('train','selection','test')}
    candidates=[]
    for seed in SEEDS:
        model,nll=train_student(seed,splits,teachers,device)
        cpu_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        candidates.append((nll,seed,cpu_state))
    selection_nll,selected_seed,state=min(candidates,key=lambda value:value[0])
    del contexts,candidates,model
    if device=='cuda':torch.cuda.empty_cache()
    model=CoalitionAwareBackbone([x.shape[1] for x in splits['train']['x']],2).to(device);model.load_state_dict(state);model.eval()
    probability=predict(model,splits['test'],device);labels=splits['test']['y'];row=np.arange(len(labels));teacher_prob=teachers['test']
    output={'selected_seed':selected_seed,'selection_nll':selection_nll,'parameters':sum(p.numel() for p in model.parameters()),
            'accuracy':float((probability.argmax(1)==labels).mean()),'nll':float(-np.log(np.clip(probability[row,labels],1e-12,1)).mean()),
            'teacher_accuracy':float((teacher_prob.argmax(1)==labels).mean()),'teacher_nll':float(-np.log(np.clip(teacher_prob[row,labels],1e-12,1)).mean())}
    rows=[]
    for batch,repeats in ((1,200),(64,100)):
        mean,std,peak=benchmark(model,splits['test'],device,batch,repeats);rows.append(output|{'batch_size':batch,'latency_ms_mean':mean,'latency_ms_std':std,'peak_vram_mb':peak})
    pd.DataFrame(rows).to_csv(ROOT/'results/s6_distilled_rcg.csv',index=False)
    torch.save({'state_dict':state,'dims':[x.shape[1] for x in splits['train']['x']],'classes':2,'selected_seed':selected_seed},ROOT/'runs/s6_distilled_mosi.pt')
    (ROOT/'results/s6_distilled_manifest.json').write_text(json.dumps({'teacher':'rcg-fusion-a8-v2','student':'single coalition backbone full-coalition distillation','student_seeds':list(SEEDS),'selection_rule':'minimum selection NLL','test_labels_role':'evaluation only'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
