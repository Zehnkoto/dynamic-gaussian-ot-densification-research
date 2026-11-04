"""4DGS loading, deformed state extraction and native covariance rendering."""
import argparse,sys,pathlib,os
import torch
from inspect_baseline import configuration

def load_scene(model):
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();model=pathlib.Path(model).resolve()
    if not model.is_relative_to(root):raise ValueError('Model outside data root')
    sys.path.insert(0,str(root/'projects/4DGaussians'))
    from arguments import ModelParams,ModelHiddenParams,PipelineParams
    from scene import Scene,GaussianModel
    cfg=configuration(model/'cfg_args')
    if pathlib.Path(cfg.source_path).resolve()!=root/'datasets/bouncingballs' or pathlib.Path(cfg.model_path).resolve()!=model:raise ValueError('Unexpected model/data path')
    if not cfg.no_do or not cfg.no_dshs:raise ValueError('This pilot requires time-shared opacity/SH')
    ap=argparse.ArgumentParser();mp=ModelParams(ap);hp=ModelHiddenParams(ap);pp=PipelineParams(ap)
    pc=GaussianModel(cfg.sh_degree,hp.extract(cfg));scene=Scene(mp.extract(cfg),pc,load_iteration=20000,shuffle=False)
    return cfg,pc,scene,pp.extract(cfg)

def state(pc,time):
    from utils.general_utils import build_rotation
    t=torch.full((len(pc.get_xyz),1),float(time),device='cuda')
    xyz,scale,rot,opacity,sh=pc._deformation(pc.get_xyz,pc._scaling,pc._rotation,pc._opacity,pc.get_features,t)
    scale=pc.scaling_activation(scale);rot=pc.rotation_activation(rot)
    R=build_rotation(rot);L=R*scale[:,None,:];cov=L@L.transpose(1,2)
    return torch.cat([xyz,cov.reshape(-1,9)],1),pc.opacity_activation(opacity),sh

def small_camera(base,size):
    from scene.cameras import Camera
    return Camera(0,base.R.copy(),base.T.copy(),base.FoVx,base.FoVy,torch.zeros(3,size,size),None,'fixed-view',0,time=0.)

def render_state(cam,features,sh,opacity,degree=3):
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    f=features.float().contiguous();cov=f[:,3:].reshape(-1,3,3)
    packed=torch.stack([cov[:,0,0],cov[:,0,1],cov[:,0,2],cov[:,1,1],cov[:,1,2],cov[:,2,2]],1).contiguous()
    settings=GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,tanfovx=math.tan(cam.FoVx/2),tanfovy=math.tan(cam.FoVy/2),bg=torch.ones(3,device='cuda'),scale_modifier=1.,viewmatrix=cam.world_view_transform.cuda(),projmatrix=cam.full_proj_transform.cuda(),sh_degree=degree,campos=cam.camera_center.cuda(),prefiltered=False,debug=False)
    image,_,_=GaussianRasterizer(settings)(means3D=f[:,:3].contiguous(),means2D=torch.zeros_like(f[:,:3]),shs=sh.float().contiguous(),colors_precomp=None,opacities=opacity.float().contiguous(),scales=None,rotations=None,cov3D_precomp=packed)
    return image.clamp(0,1)
