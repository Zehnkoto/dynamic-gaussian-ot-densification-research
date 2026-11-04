import numpy as np
from reranking_core import choose_batches
assert choose_batches([.1,-.3,.1,.2],3)==[3,0,2]
assert choose_batches([-3,-1,-2],2)==[1,2]  # Fixed budget includes negative gains.
for bad,count in [([np.nan,1],1),([np.inf],1),([[1,2]],1),([1],0),([1],2)]:
    try:choose_batches(bad,count)
    except ValueError:pass
    else:raise AssertionError((bad,count))
print('Fixed-count selection, stable ties, negative gains and invalid-score rejection passed')
