"""A second split must preserve learned first-event children and Adam history."""
import torch
from appearance_model import Appearance
from training_split_core import split_adam,RATES,digest_state

torch.manual_seed(17)
m=Appearance(torch.randn(6,16,3),torch.randn(6,1),6);opt=m.optimizer(RATES)
def update(m,opt):
 opt.zero_grad();sh,opacity=m(torch.arange(len(m.dc)));(sh.square().mean()+opacity.square().mean()).backward();opt.step()
for _ in range(3):update(m,opt)
m,opt,_=split_adam(m,opt,[1,3],[.3,.7])
for _ in range(4):update(m,opt)
signature=digest_state(m,opt);oldsh,oldop=m(torch.arange(8));new,nopt,_=split_adam(m,opt,[0,4],[.4,.6]);assert digest_state(m,opt)==signature
keep=[1,2,3,5,6,7];reset=[0,4,8,9]
for name,p in m.named_parameters():
 q=getattr(new,name);assert torch.equal(q[keep],p[keep]) and torch.count_nonzero(q[reset])==0
 for key in ['exp_avg','exp_avg_sq']:assert torch.equal(nopt.state[q][key][keep],opt.state[p][key][keep]) and torch.count_nonzero(nopt.state[q][key][reset])==0
 assert nopt.state[q]['step']==7
sh,op=new(torch.arange(10));assert torch.equal(sh[keep],oldsh[keep]) and torch.equal(op[keep],oldop[keep]);assert torch.allclose((1-op[[0,4]])*(1-op[[8,9]]),1-oldop[[0,4]],atol=1e-7)
update(new,nopt)
for p in new.parameters():assert nopt.state[p]['step']==8 and torch.isfinite(nopt.state[p]['exp_avg']).all()
print('Second-event preservation of learned children/moments, scalar age, optical depth and next update passed')
