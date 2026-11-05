import torch
from alignment_metrics import mse_decomposition

torch.manual_seed(38)
for dtype in [torch.float32,torch.float64]:
 p,t,g=[torch.rand(3,13,17,dtype=dtype) for _ in range(3)];r=mse_decomposition(p,t,g)
 assert abs(r['gt_mse']-r['teacher_mse']-r['teacher_gt_mse']-r['cross_term'])<1e-7
 r=mse_decomposition(p,g,g);assert r['teacher_mse']==r['gt_mse'] and r['teacher_gt_mse']==r['cross_term']==0
 r=mse_decomposition(t,t,g);assert r['teacher_mse']==r['cross_term']==0 and r['gt_mse']==r['teacher_gt_mse']
 # A closer teacher fit can worsen GT: the negative cross term matters.
 g=torch.zeros(1,dtype=dtype);t=torch.ones(1,dtype=dtype)
 a=mse_decomposition(g,t,g);b=mse_decomposition(t,t,g)
 assert b['teacher_mse']<a['teacher_mse'] and b['gt_mse']>a['gt_mse']
try:mse_decomposition(torch.zeros(2),torch.zeros(3),torch.zeros(2))
except ValueError:pass
else:raise AssertionError('Shape mismatch accepted')
print('MSE identity, teacher/GT limiting cases, error-cancellation sign and shape guard passed')
