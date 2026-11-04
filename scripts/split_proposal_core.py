"""Offline local two-way trajectory CTD proposals, without global reassignment."""
import numpy as np

def bipartitions(states,parents,weights,fit,max_iter=30):
    n=len(parents);k=int(parents.max())+1
    x=np.asarray(states[fit],dtype=np.float64).transpose(1,0,2).reshape(n,-1)/np.sqrt(len(fit))
    order=np.argsort(parents,kind='stable');bounds=np.r_[0,np.cumsum(np.bincount(parents,minlength=k))]
    child=np.zeros(n,dtype=np.int8);eligible=[];iterations=[]
    for p in range(k):
        ids=order[bounds[p]:bounds[p+1]]
        if len(ids)<2:continue
        a=x[ids];w=weights[ids];mean=np.average(a,axis=0,weights=w)
        first=int(np.argmax(((a-mean)**2).sum(1)));second=int(np.argmax(((a-a[first])**2).sum(1)))
        if float(((a[first]-a[second])**2).sum())<=1e-24:continue
        centers=a[[first,second]].copy();previous=None
        for it in range(max_iter):
            labels=np.argmin(((a[:,None,:]-centers[None,:,:])**2).sum(2),axis=1)
            if len(np.unique(labels))!=2:raise ValueError('Degenerate local split')
            centers=np.stack([np.average(a[labels==j],axis=0,weights=w[labels==j]) for j in [0,1]])
            if previous is not None and np.array_equal(labels,previous):break
            previous=labels.copy()
        child[ids]=labels;eligible.append(p);iterations.append(it+1)
    return child,np.asarray(eligible,dtype=np.int64),iterations

def child_trajectories(states,parents,child,weights):
    k=int(parents.max())+1;labels=parents*2+child
    mass=np.bincount(labels,weights=weights,minlength=2*k);den=np.where(mass>0,mass,1)
    centers=np.empty((len(states),2*k,12),dtype=np.float32)
    for t in range(len(states)):
        for d in range(12):centers[t,:,d]=np.bincount(labels,weights=weights*np.asarray(states[t,:,d],dtype=np.float64),minlength=2*k)/den
    a=mass[::2];b=mass[1::2];coefficient=a*b/(a+b)
    gains=coefficient[None,:]*((centers[:,::2].astype(np.float64)-centers[:,1::2])**2).sum(2)
    return centers,mass,gains

def selections(gains,eligible,fit,times,budget,seed=42):
    assert len(eligible)>=budget
    bins=np.minimum((times*5).astype(int),4)
    windows=np.stack([gains[[i for i in fit if bins[i]==b]].mean(0) for b in range(5)])
    scores={'window':windows.min(0),'global':gains[fit].mean(0),'first':gains[fit[0]]}
    selected={m:eligible[np.argsort(-score[eligible],kind='stable')[:budget]] for m,score in scores.items()}
    selected['random']=np.random.default_rng(seed).choice(eligible,budget,replace=False)
    return scores,selected,windows
