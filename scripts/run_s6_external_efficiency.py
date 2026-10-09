"""Measure unified-feature strong-baseline deployment cost on the same GPU."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.rcg_fusion_pipeline import load_dataset
from rcg.strong_models import build_model

ROOT=Path(__file__).resolve().parents[1]
METHODS=("concat","tmc","qmf","pdf","i2moe")


def count(model): return sum(p.numel() for p in model.parameters())


def benchmark(function, device, repeats):
    for _ in range(20):function()
    torch.cuda.synchronize(device);samples=[]
    for _ in range(7):
        torch.cuda.synchronize(device);start=time.perf_counter()
        for _ in range(repeats):function()
        torch.cuda.synchronize(device);samples.append((time.perf_counter()-start)*1000/repeats)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);function();torch.cuda.synchronize(device)
    return np.mean(samples),np.std(samples,ddof=1),torch.cuda.max_memory_allocated(device)/1024**2


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device('cuda');folds,names=load_dataset('mosi',ROOT/'data');split=folds[0]['test']
    dims=[x.shape[1] for x in folds[0]['train']['x']];classes=2;rows=[]
    s1=pd.read_csv(ROOT/'results/s1_clean_metrics.csv')
    for method in METHODS:
        root=ROOT/f'runs/strong-observation/mosi/fold_0/{method}/seed_11'
        metadata=json.loads((root/'run.json').read_text(encoding='utf-8'))
        model=build_model(method,dims,classes).to(device);model.load_state_dict(torch.load(root/'model.pt',map_location=device,weights_only=True));model.eval()
        for batch_size,repeats in ((1,200),(64,100)):
            idx=np.arange(batch_size)%len(split['y']);xs=[torch.as_tensor(x[idx],dtype=torch.float32,device=device) for x in split['x']]
            present=torch.ones((batch_size,len(names)),device=device)
            function=lambda:torch.softmax(model(xs,present)['logits']/metadata['temperatures']['+'.join(names)],-1)
            mean,std,peak=benchmark(function,device,repeats)
            ensemble_nll=float(s1[(s1.dataset=='mosi')&(s1.method==method)].nll.iloc[0])
            rows.append({'method':method,'batch_size':batch_size,'parameters':count(model),
                         'latency_ms_mean':mean,'latency_ms_std':std,'peak_vram_mb':peak,
                         'five_member_estimated_latency_ms':5*mean,'five_member_clean_nll':ensemble_nll})
        del model,xs,present;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'results/s6_external_efficiency.csv',index=False)


if __name__=='__main__':main()
