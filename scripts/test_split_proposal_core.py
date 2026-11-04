import numpy as np
from split_proposal_core import bipartitions,child_trajectories

def main():
    # Independently compare weighted source SSE against Ward's predicted decrease.
    rng=np.random.default_rng(7);s=rng.normal(size=(5,7,12));parents=np.array([0,0,0,1,1,1,2]);w=np.array([1,2,1,3,1,2,1],dtype=float);w/=w.sum()
    child,eligible,it=bipartitions(s,parents,w,[0,2,4]);c,m,g=child_trajectories(s,parents,child,w)
    assert set(eligible)=={0,1} and np.isclose(m.sum(),1)
    assert np.all(m[(eligible[:,None]*2+np.array([0,1])).reshape(-1)]>0)
    for t in range(5):
        for p in eligible:
            ids=np.flatnonzero(parents==p);mean=np.average(s[t,ids],axis=0,weights=w[ids]);before=(w[ids,None]*(s[t,ids]-mean)**2).sum();after=(w[ids,None]*(s[t,ids]-c[t,p*2+child[ids]])**2).sum()
            assert np.isclose(before-after,g[t,p],rtol=1e-6,atol=1e-8)
            reconstructed=(c[t,p*2]*m[p*2]+c[t,p*2+1]*m[p*2+1])/(m[p*2]+m[p*2+1]);assert np.allclose(mean,reconstructed,atol=1e-6)
    print('Local split nonempty children, mass/barycenter consistency, independent SSE gain checks passed')
if __name__=='__main__':main()
