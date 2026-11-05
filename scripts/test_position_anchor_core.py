import torch
from position_ctd_core import SharedTranslation
from position_anchor_core import uniform_relative_penalty

def main():
    k=4
    mass=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    field=SharedTranslation(mass,torch.zeros(k,3,dtype=torch.float64),.03)
    with torch.no_grad():field.raw.copy_(torch.arange(12).reshape(k,3)/10)
    d=field.offset().double()
    expected=(d*d).sum()/k/.03
    got=uniform_relative_penalty(field)
    torch.testing.assert_close(got,expected)
    g=torch.autograd.grad(got,field.raw,retain_graph=True)[0]
    torch.testing.assert_close(g,torch.autograd.grad(expected,field.raw)[0])
    assert not torch.isclose(got,field.relative_excess())
    equal=SharedTranslation(torch.full((k,),1/k,dtype=torch.float64),torch.zeros(k,3,dtype=torch.float64),.03)
    with torch.no_grad():equal.raw.copy_(field.raw)
    torch.testing.assert_close(uniform_relative_penalty(equal),equal.relative_excess())
    assert torch.isfinite(g).all() and (g!=0).any()
    print('Uniform anchor value, gradient and equal-mass equivalence passed')
if __name__=='__main__':main()
