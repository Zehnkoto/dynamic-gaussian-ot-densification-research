#!/usr/bin/env python3
"""E19 jumpingjacks shared-trajectory reduction and fixed appearance-loss arms."""
import argparse,hashlib,json,math,os,pathlib,shutil,sys,time,zipfile
import numpy as np
import torch
from PIL import Image
from appearance_model import Appearance
from covariance_moment_core import group_moments,add_scatter
from reduction_core import lloyd,barycenters
from reduction_scene import state,small_camera,render_state
from inspect_baseline import configuration

SCENE='jumpingjacks'
BASE='baseline-e19-jumpingjacks-20261110-v1'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';base=root/'outputs/dnerf'/SCENE/BASE;data=root/'datasets'/SCENE;out=args.output.resolve()
    assert out.is_relative_to(root/'outputs/ot') and not out.exists() and shutil.disk_usage(root).free>7*1024**3
    protocol_path=repo/'experiments/second-scene-validation-v1.json';protocol=json.loads(protocol_path.read_text())
    baseline=json.loads((base/'run-manifest.json').read_text())
    assert baseline['phase']=='completed' and baseline['repository_commit']==protocol['teacher']['repo_commit']
    cfg=configuration(base/'cfg_args')
    assert pathlib.Path(cfg.source_path).resolve()==data.resolve() and pathlib.Path(cfg.model_path).resolve()==base.resolve()
    assert cfg.no_do and cfg.no_dshs and cfg.coarse_iterations==3000 and cfg.iterations==20000
    model_files=sorted((base/'point_cloud/iteration_20000').glob('*'))
    assert {p.name for p in model_files}=={'point_cloud.ply','deformation.pth','deformation_table.pth','deformation_accum.pth'}
    inputs={p:digest(p) for p in [protocol_path,base/'run-manifest.json',base/'cfg_args',data/'data-manifest.json',data/'transforms_train.json',data/'transforms_test.json',*model_files]}
    for j in range(20):
        p=base/f'test/ours_20000/gt/{j:05d}.png';inputs[p]=digest(p)
    sys.path.insert(0,str(root/'projects/4DGaussians'))
    from arguments import ModelParams,ModelHiddenParams,PipelineParams
    from scene import Scene,GaussianModel
    ap2=argparse.ArgumentParser();mp=ModelParams(ap2);hp=ModelHiddenParams(ap2);pp=PipelineParams(ap2)
    pc=GaussianModel(cfg.sh_degree,hp.extract(cfg));scene=Scene(mp.extract(cfg),pc,load_iteration=20000,shuffle=False);pipe=pp.extract(cfg)
    for x in vars(pc).values():
        if isinstance(x,torch.nn.Parameter):x.requires_grad_(False)
        elif isinstance(x,torch.nn.Module):x.requires_grad_(False);x.eval()
    n=len(pc.get_xyz);k=round(n*10000/26936)
    assert 0<k<n
    frames=json.loads((data/'transforms_train.json').read_text())['frames'];times=np.array(sorted(r['time'] for r in frames),dtype=np.float64)
    assert len(times)==len(set(times))==200
    fit=list(range(0,200,2))+[199];held=list(range(1,199,2));assert len(fit)==101 and len(held)==99
    rng=np.random.default_rng(42);orders=[rng.permutation(fit).tolist() for _ in range(20)]
    out.mkdir(parents=True);started=time.monotonic()
    dump(out/'manifest.json',{'protocol':protocol,'baseline':str(base),'n':n,'k':k,'retention':k/n,'fit_indices':fit,'holdout_indices':held,'orders':orders,'inputs':{str(p):h for p,h in inputs.items()},'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')})
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['run_second_scene_validation.py','appearance_model.py','covariance_moment_core.py','reduction_core.py','reduction_scene.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/name,'scripts/'+name)
        z.write(protocol_path,'experiments/'+protocol_path.name)
    def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
    with torch.no_grad():
        f0,o0,sh0=state(pc,0.)
        cache=np.lib.format.open_memmap(out/'teacher-states.npy',mode='w+',dtype=np.float32,shape=(200,n,12))
        for i,t in enumerate(times):
            f,o,sh=state(pc,float(t));assert torch.equal(o,o0) and torch.equal(sh,sh0)
            cache[i]=f.cpu().numpy()
            if i%50==0:print('TEACHER STATES',i,flush=True)
        cache.flush();del cache
    full=np.load(out/'teacher-states.npy',mmap_mode='r')
    sh=sh0.detach();opacity=o0.detach();logits=pc._opacity.detach();weights=opacity.flatten().double();weights/=weights.sum()
    fitting=torch.tensor(np.transpose(full[fit],(1,0,2)).copy().reshape(n,-1),device='cuda',dtype=torch.float64)/math.sqrt(len(fit))
    initial_ids=torch.randperm(n,generator=torch.Generator().manual_seed(42))[:k].cuda()
    shared=lloyd(fitting,weights,initial_ids,max_iter=20);labels=shared['labels'];reps=shared['representatives'];del fitting
    print('SHARED GROUP FIT',json.dumps(shared['info']),flush=True)
    np.savez(out/'shared-grouping.npz',labels=labels.cpu().numpy(),representatives=reps.cpu().numpy(),normalized_mass=shared['mass'].cpu().numpy(),initial_source_ids=initial_ids.cpu().numpy())
    mean=np.lib.format.open_memmap(out/'mean-states.npy',mode='w+',dtype=np.float32,shape=(200,k,12))
    moment=np.lib.format.open_memmap(out/'moment-states.npy',mode='w+',dtype=np.float32,shape=(200,k,12))
    lab=labels.cpu().numpy();w=weights.cpu().numpy();max_mu=0.;max_cov=0.
    with torch.no_grad():
        for i in range(200):
            x=tensor(full[i]).double();center,_=barycenters(x,labels,weights,k)
            a=center.float().cpu().numpy();mean[i]=a
            mu,cov,scatter,mass=group_moments(full[i],lab,w,k)
            max_mu=max(max_mu,float(np.max(abs(a[:,:3]-mu))))
            max_cov=max(max_cov,float(np.max(abs(a[:,3:].reshape(k,3,3)-cov))))
            moment[i]=add_scatter(a,scatter,1)
            if i%50==0:print('REPRESENTATIVE STATES',i,flush=True)
    mean.flush();moment.flush();del mean,moment
    assert max_mu<1e-6 and max_cov<1e-6,(max_mu,max_cov)
    dump(out/'geometry-audit.json',{'source_group_mean_max_abs_error':max_mu,'source_mean_covariance_max_abs_error':max_cov,'shared_fit':shared['info'],'positions_and_labels_fixed_across_four_arms':True})
    features={'mean_covariance':np.load(out/'mean-states.npy',mmap_mode='r'),'moment_covariance':np.load(out/'moment-states.npy',mmap_mode='r')}
    train=scene.getTrainCameras();cameras={}
    for j,r in enumerate(frames):
        i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7 and abs(float(train[j].time)-r['time'])<1e-7
        cameras[i]=small_camera(train[j],400)
    assert len(cameras)==200
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    import lpips
    criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    def pixel_loss(p,t):return .8*(p-t).abs().mean()+.2*(1-ssim(p.unsqueeze(0),t.unsqueeze(0)))
    def render(condition,i,cam,model):
        colors,alpha=model(reps)
        return render_state(cam,tensor(features[condition][i]),colors,alpha,cfg.sh_degree)
    teacher={}
    with torch.no_grad():
        for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
    models={};logs=[];hashes={}
    for condition in ['mean_covariance','moment_covariance']:
        for beta in [0.,0.1]:
            name=f'{condition}__beta{beta:g}';model=Appearance(sh,logits,k);optimizer=model.optimizer({'dc':0.0025,'rest':0.000125,'opacity':0.01})
            for epoch,order in enumerate(orders):
                losses=[]
                for i in order:
                    optimizer.zero_grad(set_to_none=True);pred=render(condition,i,cameras[i],model)
                    loss=pixel_loss(pred,teacher[i])
                    if beta:loss=loss+beta*criterion(pred.unsqueeze(0),teacher[i].unsqueeze(0),normalize=True).mean()
                    assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss.detach()))
                row={'arm':name,'epoch':epoch+1,'updates':(epoch+1)*101,'fit_loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started};logs.append(row)
                (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
                if (epoch+1)%5==0:print('TRAIN',json.dumps(row),flush=True)
            assert all(int(x['step'])==2020 for x in optimizer.state.values())
            path=out/(name+'-appearance.pt');torch.save({'state_dict':{q:v.detach().cpu() for q,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'steps':2020,'condition':condition,'beta':beta},path)
            hashes[path.name]=digest(path);models[name]=(condition,model)
    dump(out/'training.json',logs);dump(out/'training-complete.json',{'all4_frozen_before_evaluation':True,'checkpoint_hashes':hashes,'new_gradient_updates':8080})
    print('ALL FOUR FROZEN; EVALUATING',flush=True);teacher.clear()
    gtrows=[];fidelity=[];temporal=[];previous=None
    with torch.no_grad():
        tests=json.loads((data/'transforms_test.json').read_text())['frames']
        for j,r in enumerate(tests):
            i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7
            gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
            for name,(condition,model) in models.items():
                pred=(render(condition,i,scene.getTestCameras()[j],model)*255).round()/255;p=pred.unsqueeze(0)
                gtrows.append({'arm':name,'view':j,'PSNR':float(psnr(p,gt)),'SSIM':float(ssim(p,gt)),'LPIPS_alex_v01':float(criterion(p,gt,normalize=True))})
        print('GT COMPLETE',flush=True)
        fixed=small_camera(scene.getTestCameras()[10],256)
        for i in range(200):
            target=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree)
            images={'teacher':render_state(fixed,tensor(full[i]),sh,opacity,cfg.sh_degree)}
            for name,(condition,model) in models.items():
                p=render(condition,i,cameras[i],model)
                fidelity.append({'arm':name,'time_index':i,'split':'fit' if i in fit else 'holdout','PSNR_to_teacher':float(psnr(p.unsqueeze(0),target.unsqueeze(0))),'L1_to_teacher':float((p-target).abs().mean()),'mixed_loss_to_teacher':float(pixel_loss(p,target))})
                images[name]=render(condition,i,fixed,model)
            if previous is not None:
                for name in models:
                    temporal.append({'arm':name,'time_index':i,'residual':float(((images[name]-previous[name])-(images['teacher']-previous['teacher'])).abs().mean())})
            previous=images
            if i%50==0:print('FIDELITY/TEMPORAL',i,flush=True)
    dump(out/'gt-per-view.json',gtrows);dump(out/'fidelity-per-view.json',fidelity);dump(out/'temporal-per-pair.json',temporal)
    rows=[]
    for name in models:
        gt=[r for r in gtrows if r['arm']==name];tmp=[r['residual'] for r in temporal if r['arm']==name]
        row={'arm':name,**{m:float(np.mean([r[m] for r in gt])) for m in ['PSNR','SSIM','LPIPS_alex_v01']},'temporal_residual256':float(np.mean(tmp))}
        for split in ['fit','holdout']:
            vals=[r for r in fidelity if r['arm']==name and r['split']==split]
            row[split]={m:float(np.mean([r[m] for r in vals])) for m in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']}
        rows.append(row)
    by={r['arm']:r for r in rows};metrics=['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']
    paired_beta={c:{m:by[f'{c}__beta0.1'][m]-by[f'{c}__beta0'][m] for m in metrics} for c in features}
    paired_shape={b:{m:by[f'moment_covariance__beta{b:g}'][m]-by[f'mean_covariance__beta{b:g}'][m] for m in metrics} for b in [0.,0.1]}
    dump(out/'report.json',{'teacher_gaussians':n,'shared_budget':k,'retention':k/n,'rows':rows,'paired_beta01_minus_beta0':paired_beta,'paired_moment_minus_mean':paired_shape,'scope':protocol['scope']})
    for p,h in inputs.items():assert digest(p)==h,str(p)
    for name,h in hashes.items():assert digest(out/name)==h,name
    dump(out/'completion.json',{'status':'completed','inputs_unchanged':True,'checkpoints_verified':True,'source_sha256':digest(out/'source.zip'),'seconds':time.monotonic()-started})
    print('RESULT',json.dumps({'paired_beta':paired_beta,'paired_shape':paired_shape,'rows':rows}),flush=True)

if __name__=='__main__':main()
