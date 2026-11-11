"""Moment-matched covariance diagnostic under fixed source assignments."""
import numpy as np

def group_moments(features, labels, weights, k):
    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.shape != (len(labels),12) or weights.shape != labels.shape:
        raise ValueError("Invalid shapes")
    if not np.isfinite(x).all() or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError("Nonfinite states or nonpositive weights")
    if labels.min()<0 or labels.max()>=k:
        raise ValueError("Invalid labels")
    mass = np.bincount(labels, weights=weights, minlength=k)
    if (mass<=0).any():
        raise ValueError("Empty source group")
    mean = np.stack([np.bincount(labels,weights=weights*x[:,j],minlength=k)/mass for j in range(3)],axis=1)
    delta=x[:,:3]-mean[labels]
    scatter=np.empty((k,3,3),dtype=np.float64)
    cov=np.empty_like(scatter)
    for j in range(3):
        for l in range(3):
            scatter[:,j,l]=np.bincount(labels,weights=weights*delta[:,j]*delta[:,l],minlength=k)/mass
            cov[:,j,l]=np.bincount(labels,weights=weights*x[:,3+3*j+l],minlength=k)/mass
    return mean,cov,scatter,mass

def add_scatter(base, scatter, coefficient):
    base=np.asarray(base)
    scatter=np.asarray(scatter)
    if base.shape[-1]!=12 or scatter.shape!=base.shape[:-1]+(3,3):
        raise ValueError("Invalid shapes")
    if coefficient not in (0,1):
        raise ValueError("Only the two predeclared constructions are allowed")
    result=np.array(base,copy=True)
    if coefficient:
        result[...,3:]=(base[...,3:].astype(np.float64)+scatter.reshape(base.shape[:-1]+(9,))).astype(base.dtype)
    return result
