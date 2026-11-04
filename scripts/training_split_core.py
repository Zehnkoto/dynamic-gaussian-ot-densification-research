"""Pure appearance/Adam state migration and fit-only render gate for one split event."""
import copy,hashlib,json
import numpy as np
import torch
from appearance_model import Appearance
RATES={'dc':.0025,'rest':.000125,'opacity':.01}

def clone_adam(model,optimizer):
    result=Appearance(model.source_sh,model.source_logits,len(model.dc))
    result.load_state_dict(model.state_dict());opt=result.optimizer(RATES)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return result,opt

@torch.no_grad()
def split_adam(model,optimizer,parents,fractions):
    device=model.dc.device;k=len(model.dc);ids=torch.as_tensor(parents,device=device,dtype=torch.long);b=len(ids)
    assert b>0 and len(torch.unique(ids))==b and bool(((ids>=0)&(ids<k)).all())
    f=torch.as_tensor(fractions,device=device,dtype=torch.float64).reshape(-1,1)
    assert len(f)==b and bool(((f>0)&(f<1)).all())
    effective_sh=model.source_sh+torch.cat([model.dc,model.rest],1);effective_logits=model.source_logits+model.opacity
    sh=torch.cat([model.source_sh,effective_sh[ids]],0).clone();logits=torch.cat([model.source_logits,effective_logits[ids]],0).clone();sh[ids]=effective_sh[ids]
    tau=torch.nn.functional.softplus(effective_logits[ids].double())
    logits[ids]=torch.log(torch.expm1(tau*f)).float();logits[k:]=torch.log(torch.expm1(tau*(1-f))).float()
    result=Appearance(sh,logits,k+b)
    for name,p in result.named_parameters():
        p[:k].copy_(getattr(model,name));p[ids]=0
    opt=result.optimizer(RATES)
    for oldgroup,newgroup in zip(optimizer.param_groups,opt.param_groups):
        for key,value in oldgroup.items():
            if key!='params':newgroup[key]=copy.deepcopy(value)
    for name,newparam in result.named_parameters():
        oldparam=getattr(model,name);state={}
        for key,value in optimizer.state.get(oldparam,{}).items():
            if torch.is_tensor(value) and value.shape==oldparam.shape:
                expanded=torch.zeros_like(newparam);expanded[:k]=value;expanded[ids]=0;state[key]=expanded
            else:state[key]=value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        if state:opt.state[newparam]=state
    error=float((torch.nn.functional.softplus(logits[ids].double())+torch.nn.functional.softplus(logits[k:].double())-tau).abs().max())
    assert error<1e-5
    return result,opt,error

def digest_state(model,optimizer):
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            a=value.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):h.update(str(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            for v in value:visit(v)
        else:h.update(repr(value).encode())
    visit(model.state_dict());visit(optimizer.state_dict());return h.hexdigest()

def render_gate(reference_losses,proposal_losses,window_ids,count,price=1e-7,max_worsening=.01):
    ref=np.asarray(reference_losses,dtype=np.float64);prop=np.asarray(proposal_losses,dtype=np.float64);bins=np.asarray(window_ids)
    assert ref.shape==prop.shape==bins.shape and np.isfinite(ref).all() and np.isfinite(prop).all()
    gain=float((ref-prop).mean());penalty=float(count*price)
    by_window=[{'window':b,'reference':float(ref[bins==b].mean()),'proposal':float(prop[bins==b].mean())} for b in range(5)]
    worst=max((r['proposal']-r['reference'])/max(r['reference'],1e-12) for r in by_window)
    return {'accept':bool(gain>penalty and worst<=max_worsening),'render_gain':gain,'complexity_penalty':penalty,'net_gain':gain-penalty,'worst_window_relative_worsening':float(worst),'windows':by_window}
