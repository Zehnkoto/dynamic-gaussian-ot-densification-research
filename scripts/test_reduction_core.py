"""Numerical parity against pinned GHAP plus exact-budget and mass checks."""
import importlib.util,os,pathlib
import torch
from reduction_core import lloyd,nearest,barycenters

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    root=pathlib.Path(os.environ['RESEARCH_ROOT']);source=root/'projects/GHAP/gaussian_splatting_sampler/gmr.py'
    spec=importlib.util.spec_from_file_location('reference',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    means=torch.randn(53,3,dtype=torch.float64);a=torch.randn(53,3,3,dtype=torch.float64)*.02
    cov=a@a.transpose(1,2)+torch.eye(3,dtype=torch.float64)*.001
    weights=torch.rand(53,dtype=torch.float64)+.1;weights/=weights.sum();ids=torch.randperm(53)[:7]
    x=torch.cat([means,cov.reshape(53,9)],1)
    expected=ref.TensorKL([means,means[ids]],[cov,cov[ids]])
    direct=((x[:,None]-x[ids][None])**2).sum(2)
    assert torch.allclose(expected,direct,atol=1e-12,rtol=1e-12)
    labels,errors=nearest(x,x[ids],block=9)
    assert torch.equal(labels,expected.argmin(1)) and torch.allclose(errors,expected.min(1).values,atol=1e-12)
    original=ref.GMM_Sampler.__new__(ref.GMM_Sampler)
    original.means=means;original.covs=cov;original.weights=weights;original.cost_matrix=expected
    original._weight_update();original._support_update()
    result=lloyd(x,weights,ids,max_iter=1,block=9)
    assert torch.allclose(result['centers'][:,:3],original.reduced_means,atol=1e-12)
    assert torch.allclose(result['centers'][:,3:].reshape(7,3,3),original.reduced_covs,atol=1e-12)
    assert torch.allclose(result['mass'],original.reduced_weights,atol=1e-12)
    assert torch.equal(result['representatives'],original.cost_matrix.argmin(0))
    assert abs(result['mass'].sum().item()-1)<1e-12
    assert (torch.linalg.eigvalsh(result['centers'][:,3:].reshape(7,3,3))>0).all()
    identity=lloyd(x,weights,torch.arange(53),max_iter=3,block=9)
    assert identity['info']['objective_history'][-1]<1e-25
    duplicate=lloyd(torch.zeros(12,4,dtype=torch.float64),torch.ones(12),torch.arange(5),max_iter=3,block=4)
    assert len(torch.unique(duplicate['labels']))==5 and (duplicate['mass']>0).all()
    assert duplicate['info']['empty_cluster_repairs']>0
    shared=lloyd(x.reshape(53,1,12).flatten(1),weights,ids,max_iter=5,block=9)
    frame=lloyd(x,weights,ids,max_iter=5,block=9)
    assert torch.equal(shared['labels'],frame['labels'])
    print('PASS: GHAP cost, weight/support update, representative parity; mass; positive covariance; identity; duplicate exact budget; T=1 equivalence.')
if __name__=='__main__':main()
