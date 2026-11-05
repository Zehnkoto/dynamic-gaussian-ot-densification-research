"""Compare the quadratic to direct source-wise CTD, including nonzero centroid residuals."""
import numpy as np
import torch
from position_ctd_core import source_partition,fixed_plan_stats,SharedTranslation,apply_translation
rng=np.random.default_rng(87);states=rng.normal(size=(4,6,12));labels=np.array([0,0,1,1,2,2]);weights=np.array([.1,.2,.05,.15,.3,.2]);geometry=np.stack([np.stack([np.average(x[labels==j],axis=0,weights=weights[labels==j]) for j in range(3)]) for x in states]);geometry[:,:,:3]+=.03 # Nonzero linear term must be retained.
s=fixed_plan_stats(states,geometry,labels,weights,[0,2],[1,3]);delta=rng.normal(size=(3,3))*.01
for indices,key in [([0,2],'fit'),([1,3],'held')]:
 direct=[]
 for t in indices:
  old=(weights[:,None]*(states[t]-geometry[t,labels])**2).sum();new=geometry[t].copy();new[:,:3]+=delta;direct.append((weights[:,None]*(states[t]-new[labels])**2).sum()-old)
 quadratic=(s['mass'][:,None]*delta**2+2*s[key+'_linear']*delta).sum();assert abs(np.mean(direct)-quadratic)<1e-12
field=SharedTranslation(torch.tensor(s['mass']),torch.tensor(s['fit_linear']),s['fit_base']);assert torch.count_nonzero(field.offset())==0
field.relative_excess().backward();assert torch.isfinite(field.raw.grad).all() and field.raw.grad.abs().sum()>0
with torch.no_grad():field.raw.copy_(torch.tensor(rng.normal(size=(3,3)),dtype=torch.float32))
d=field.offset().double();direct=torch.stack([(torch.tensor(weights)[:,None]*(torch.tensor(states[t,:,:3])-torch.tensor(geometry[t,labels,:3])-d[torch.tensor(labels)])**2).sum() for t in [0,2]]).mean()/s['fit_base']
grad_direct=torch.autograd.grad(direct,field.raw,retain_graph=True)[0];grad_quad=torch.autograd.grad(field.relative_excess(),field.raw)[0];assert torch.allclose(grad_direct,grad_quad,atol=1e-7,rtol=1e-5)
with torch.no_grad():field.raw.fill_(10000)
assert float(field.offset().norm(dim=1).max())<=.01000001
f=torch.tensor(geometry[0],dtype=torch.float32);g=torch.tensor(geometry[1],dtype=torch.float32);d=field.offset();a=apply_translation(f,d);b=apply_translation(g,d);assert torch.equal(a[:,3:],f[:,3:]) and torch.allclose(b[:,:3]-a[:,:3],g[:,:3]-f[:,:3],atol=1e-6)
part=source_partition([0,0,1,1],[0,1,0,1],[1,0]);assert part.tolist()==[0,3,1,2]
try:source_partition([0,0],[0,1],[0,0])
except ValueError:pass
else:raise AssertionError('Duplicate parents accepted')
print('Direct CTD delta equality, residual linear term, finite gradients, radius bound, fixed motion/covariance and partition IDs passed')
