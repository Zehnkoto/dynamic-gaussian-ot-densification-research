"""Bounded shared translations and exact fixed-hard-plan CTD sufficient statistics."""
import numpy as np
import torch

def source_partition(parent_labels,source_child,split_parents):
    labels=np.array(parent_labels,dtype=np.int64,copy=True);source_child=np.asarray(source_child,dtype=np.int8);original=labels.copy();k=int(labels.max())+1
    if source_child.shape!=labels.shape or not np.isin(source_child,[0,1]).all():raise ValueError('Invalid child labels')
    if len(set(split_parents))!=len(split_parents):raise ValueError('Duplicate split parents')
    for j,p in enumerate(split_parents):labels[(original==p)&(source_child==1)]=k+j
    if len(np.unique(labels))!=k+len(split_parents):raise ValueError('Empty fixed-plan support')
    return labels

def fixed_plan_stats(states,geometry,labels,weights,fit,held):
    weights=np.asarray(weights,dtype=np.float64);weights=weights/weights.sum();k=geometry.shape[1];mass=np.bincount(labels,weights=weights,minlength=k)
    if len(mass)!=k or np.any(mass<=0):raise ValueError('Invalid support mass')
    base=[];linear=[]
    for t in range(len(states)):
        x=np.asarray(states[t],dtype=np.float64);g=np.asarray(geometry[t],dtype=np.float64);base.append(float((weights[:,None]*(x-g[labels])**2).sum()))
        sums=np.stack([np.bincount(labels,weights=weights*x[:,d],minlength=k) for d in range(3)],1);linear.append(mass[:,None]*g[:,:3]-sums)
    base=np.array(base);linear=np.stack(linear)
    return {'mass':mass,'fit_linear':linear[fit].mean(0),'held_linear':linear[held].mean(0),'base_ctd':base,'fit_base':float(base[fit].mean()),'held_base':float(base[held].mean())}

def apply_translation(features,offset):
    if features.shape!=(len(offset),12) or offset.shape[1:]!=(3,):raise ValueError('Invalid geometry shape')
    return torch.cat([features[:,:3]+offset,features[:,3:]],1)

class SharedTranslation(torch.nn.Module):
    def __init__(self,mass,linear,baseline,radius=.01):
        super().__init__();self.radius=float(radius)
        if not self.radius>0:raise ValueError('Positive radius required')
        self.register_buffer('mass',mass.detach().double().clone());self.register_buffer('linear',linear.detach().double().clone());self.register_buffer('baseline',torch.as_tensor(baseline,dtype=torch.float64,device=mass.device).detach().clone())
        if not bool((self.mass>0).all()) or not bool(self.baseline>0) or self.linear.shape!=(len(mass),3):raise ValueError('Invalid CTD statistics')
        self.raw=torch.nn.Parameter(torch.zeros(len(mass),3,device=mass.device))
    def offset(self):return self.radius*self.raw/torch.sqrt(1+self.raw.square().sum(1,keepdim=True))
    def excess(self):
        d=self.offset().double();return (self.mass[:,None]*d.square()+2*self.linear*d).sum()
    def relative_excess(self):return self.excess()/self.baseline
