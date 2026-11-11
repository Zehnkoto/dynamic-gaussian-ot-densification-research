"""Verify probability moments and isolate covariance-only changes."""
import numpy as np
from covariance_moment_core import group_moments,add_scatter

def main():
    # Two narrow Gaussians at +/-2 need variance 4.25 after moment matching.
    means=np.array([[-2.,0,0],[2.,0,0],[5.,1,0]])
    cov=np.repeat((np.eye(3)*.25)[None],3,axis=0)
    features=np.concatenate([means,cov.reshape(3,9)],axis=1)
    labels=np.array([0,0,1]);weights=np.array([1.,1.,2.])
    mu,avg,scatter,mass=group_moments(features,labels,weights,2)
    base=np.concatenate([mu,avg.reshape(2,9)],axis=1).astype(np.float32)
    assert np.array_equal(add_scatter(base,scatter,0),base)
    out=add_scatter(base,scatter,1)
    assert np.array_equal(out[:,:3],base[:,:3])
    assert abs(out[0,3]-4.25)<1e-12
    assert np.array_equal(scatter[1],np.zeros((3,3)))
    for g in range(2):
        ids=np.where(labels==g)[0];p=weights[ids]/mass[g]
        expected=sum(q*(cov[i]+np.outer(means[i],means[i])) for i,q in zip(ids,p))
        actual=out[g,3:].reshape(3,3)+np.outer(out[g,:3],out[g,:3])
        assert np.allclose(expected,actual,atol=1e-12)
    rng=np.random.default_rng(12);n=91;k=7
    A=rng.normal(size=(n,3,3));cov=A@A.transpose(0,2,1)
    means=rng.normal(size=(n,3));x=np.concatenate([means,cov.reshape(n,9)],1)
    labels=np.arange(n)%k;w=rng.uniform(.01,2,n)
    mu,avg,scatter,mass=group_moments(x,labels,w,k)
    for g in range(k):
        take=labels==g;p=w[take]/mass[g];d=means[take]-mu[g]
        assert np.allclose(scatter[g],np.einsum('n,ni,nj->ij',p,d,d),atol=1e-12)
        assert np.linalg.eigvalsh(avg[g]+scatter[g]).min()>0
    print("PASS: zero control exact; positions unchanged; total second moment conserved; singleton scatter zero; weighted direct covariance parity and PSD.")
if __name__=='__main__':main()
