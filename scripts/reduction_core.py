"""Memory-bounded hard-transport CTD adaptation of pinned GHAP core.
Cost is squared mean distance plus squared Frobenius covariance distance,
not KL divergence and not Gaussian W2. Shared trajectories concatenate time
features / sqrt(T). Full geometry uses float64 for distance cancellation.
"""
import time
import numpy as np
import torch

def nearest(x,centers,block=1024):
    norm=(centers*centers).sum(1);labels=[];errors=[]
    for start in range(0,len(x),block):
        a=x[start:start+block]
        cost=torch.mm(a,centers.T).mul_(-2).add_((a*a).sum(1,keepdim=True)).add_(norm[None,:]).clamp_min_(0)
        d,j=cost.min(1);labels.append(j);errors.append(d)
    return torch.cat(labels),torch.cat(errors)

def repair_empty(labels,errors,weights,k):
    counts=torch.bincount(labels,minlength=k);empty=torch.where(counts==0)[0]
    if not len(empty):return labels,0
    # Reassign a high-error member of a non-singleton cluster; preserve exactly K.
    lab=labels.cpu().numpy().copy();cnt=counts.cpu().numpy().copy()
    priority=np.argsort(-(errors*weights).cpu().numpy(),kind='stable');cursor=0
    for target in empty.cpu().tolist():
        while cnt[lab[priority[cursor]]]<=1:cursor+=1
        idx=priority[cursor];cursor+=1;cnt[lab[idx]]-=1;lab[idx]=target;cnt[target]+=1
    return torch.from_numpy(lab).to(labels.device),len(empty)

def barycenters(x,labels,weights,k):
    mass=torch.zeros(k,dtype=x.dtype,device=x.device);mass.index_add_(0,labels,weights)
    if not (mass>0).all():raise ValueError('Empty cluster')
    centers=torch.zeros((k,x.shape[1]),dtype=x.dtype,device=x.device)
    centers.index_add_(0,labels,x*weights[:,None]);centers/=mass[:,None]
    return centers,mass

def representatives(x,centers,labels):
    # GHAP chooses the nearest original source to each support globally.
    return nearest(centers,x,block=512)[0]

def lloyd(x,weights,initial_ids,max_iter=20,tol=1e-5,block=1024):
    x=x.to(torch.float64);weights=weights.to(device=x.device,dtype=x.dtype)
    if not torch.isfinite(x).all() or not torch.isfinite(weights).all() or not (weights>0).all():raise ValueError('Nonfinite/nonpositive input')
    weights=weights/weights.sum();k=len(initial_ids)
    if not 1<=k<=len(x) or len(torch.unique(initial_ids))!=k:raise ValueError('Invalid initial supports')
    centers=x[initial_ids].clone();history=[];repairs=0;converged=False
    if x.is_cuda:torch.cuda.synchronize()
    started=time.perf_counter()
    for iteration in range(max_iter):
        labels,errors=nearest(x,centers,block);labels,n=repair_empty(labels,errors,weights,k);repairs+=n
        centers,mass=barycenters(x,labels,weights,k)
        objective=float((weights*((x-centers[labels])**2).sum(1)).sum().item())
        if history and objective>history[-1]*(1+1e-7)+1e-12:raise ValueError('CTD objective increased')
        history.append(objective)
        if len(history)>1 and abs(history[-2]-objective)<=tol*max(history[-2],1e-12):converged=True;break
    reps=representatives(x,centers,labels)
    if x.is_cuda:torch.cuda.synchronize()
    return {'labels':labels,'centers':centers,'mass':mass,'representatives':reps,'info':{'seconds':time.perf_counter()-started,'iterations':len(history),'converged':converged,'objective_history':history,'empty_cluster_repairs':repairs,'count':k}}
