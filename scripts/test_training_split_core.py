"""Check migration against independent state/value invariants, including rollback."""
import numpy as np
import torch
from appearance_model import Appearance
from training_split_core import clone_adam,split_adam,digest_state,render_gate,RATES

def main():
    torch.manual_seed(3);m=Appearance(torch.randn(5,16,3),torch.randn(5,1),5);opt=m.optimizer(RATES);ids=torch.arange(5)
    for _ in range(3):
        opt.zero_grad();sh,op=m(ids);(sh.square().mean()+op.square().mean()).backward();opt.step()
    signature=digest_state(m,opt);copy,copt=clone_adam(m,opt)
    assert digest_state(copy,copt)==signature
    split,sopt,error=split_adam(m,opt,[1,3],[.3,.7]);assert digest_state(m,opt)==signature and len(split.dc)==7
    untouched=[0,2,4]
    for name,p in m.named_parameters():
        q=getattr(split,name);assert torch.equal(p[untouched],q[untouched]);assert torch.count_nonzero(q[[1,3,5,6]])==0
        for key in ['exp_avg','exp_avg_sq']:
            assert torch.equal(opt.state[p][key][untouched],sopt.state[q][key][untouched]);assert torch.count_nonzero(sopt.state[q][key][[1,3,5,6]])==0
        assert opt.state[p]['step']==sopt.state[q]['step']
    old_sh,old_op=m(ids);new_sh,new_op=split(torch.arange(7));assert torch.equal(old_sh[1],new_sh[1]) and torch.equal(old_sh[1],new_sh[5])
    assert torch.allclose((1-new_op[[1,3]])*(1-new_op[[5,6]]),1-old_op[[1,3]],atol=1e-7)
    for model,optimizer in [(m,opt),(copy,copt)]:
        optimizer.zero_grad();a,b=model(ids);(a.sum()+b.sum()).backward();optimizer.step()
    assert digest_state(m,opt)==digest_state(copy,copt)
    bins=np.repeat(np.arange(5),4);reference=np.full(20,.01)
    assert not render_gate(reference,reference,bins,16)['accept']
    assert render_gate(reference,reference-.001,bins,16)['accept']
    bad=reference-.001;bad[:4]=reference[:4]*1.02;assert not render_gate(reference,bad,bins,16)['accept']
    print('Adam clone/next-update equality, untouched moments, reset child moments, optical depth and reject/accept guards passed')
if __name__=='__main__':main()
