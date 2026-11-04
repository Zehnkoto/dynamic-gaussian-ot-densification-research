"""Deterministic fixed-budget batch selection from fit-only scores."""
import numpy as np

def choose_batches(scores,count):
    a=np.asarray(scores,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all() or not isinstance(count,int) or not 0<count<=len(a):
        raise ValueError('Expected finite 1D scores and valid fixed count')
    return np.argsort(-a,kind='stable')[:count].tolist()
