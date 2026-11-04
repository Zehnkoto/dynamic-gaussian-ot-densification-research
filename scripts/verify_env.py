#!/usr/bin/env python3
"""Small real GPU checks for both locally compiled extensions; no training."""
import json
import torch
import numpy as np
import diff_gaussian_rasterization
from simple_knn._C import distCUDA2

assert torch.__version__.startswith('1.13.1'), 'Unexpected PyTorch version'
assert torch.version.cuda == '11.7', 'Unexpected PyTorch CUDA build'
assert torch.cuda.is_available(), 'CUDA unavailable'
# Training reads NumPy images/point clouds; validate both directions of the bridge.
array = np.array([1., 2.], dtype=np.float32)
roundtrip = torch.from_numpy(array).cuda().cpu().numpy()
assert np.array_equal(array, roundtrip), 'NumPy/PyTorch interoperability failed'
# Exercise the same histogram serialization used by the final training report.
from torch.utils.tensorboard.summary import histogram
histogram('validation/opacity', torch.tensor([0.1, 0.5, 0.9]), bins=10)
points = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], device='cuda')
dist = distCUDA2(points)
assert dist.shape == (4,) and torch.isfinite(dist).all() and (dist > 0).all()
# Exercise renderer forward/backward with a visible Gaussian on an identity camera.
R = diff_gaussian_rasterization
settings = R.GaussianRasterizationSettings(
    image_height=16, image_width=16, tanfovx=1., tanfovy=1.,
    bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
    viewmatrix=torch.eye(4, device='cuda'), projmatrix=torch.eye(4, device='cuda'),
    sh_degree=0, campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False)
xyz = torch.tensor([[0., 0., 0.5]], device='cuda', requires_grad=True)
screen = torch.zeros_like(xyz, requires_grad=True)
color = torch.tensor([[1.,0.3,0.2]], device='cuda', requires_grad=True)
image, radii, depth = R.GaussianRasterizer(settings)(
    means3D=xyz, means2D=screen, colors_precomp=color, shs=None,
    opacities=torch.full((1,1), .5, device='cuda'),
    scales=torch.full((1,3), .1, device='cuda'),
    rotations=torch.tensor([[1.,0.,0.,0.]], device='cuda'), cov3D_precomp=None)
assert (radii > 0).any() and torch.isfinite(image).all() and image.sum() > 0
image.sum().backward()
assert color.grad is not None and torch.isfinite(color.grad).all()
torch.cuda.synchronize()
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(0), 'numpy': np.__version__,
                  'numpy_roundtrip': 'passed', 'tensorboard_histogram': 'passed', 'knn': dist.tolist(),
                  'raster_forward_backward': 'passed'}, indent=2))
