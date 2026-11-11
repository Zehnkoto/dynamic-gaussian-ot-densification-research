#!/usr/bin/env python3
"""E18: fixed teacher-perceptual fit loss, unchanged E16 geometry and training orders."""
import argparse,json,os,pathlib,time,zipfile,shutil,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw
from appearance_model import Appearance
from covariance_moment_core import group_moments,add_scatter
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=args.output.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('Unsafe or existing output')
    assert shutil.disk_usage(root).free>3*1024**3
    protocol_path=repo/'experiments/perceptual-fit-loss-v1.json';protocol=json.loads(protocol_path.read_text());k=protocol['budget']
    base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z'
    cache=root/'outputs/ot/e02-ctd-20261103T082303Z/trajectory-cache'
    prior=root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000'
    e04=root/'outputs/ot/e04-appearance-20261103T095355Z'
    e16=root/'outputs/ot/e16-covariance-moment-diagnostic-20261110-v1'
    controls=json.loads((e16/'report.json').read_text())
    control_completion=json.loads((e16/'completion.json').read_text())
    control_training=json.loads((e16/'training-complete.json').read_text())
    assert control_completion['status']=='completed'
    old_manifest=json.loads((e04/'manifest.json').read_text())
    inputs={pathlib.Path(p):h for p,h in old_manifest['inputs'].items()}
    for path in [protocol_path,e16/'report.json',e16/'completion.json',e16/'training-complete.json',e16/'gt-per-view.json',e16/'fidelity-per-view.json',e16/'temporal-per-pair.json',e04/'manifest.json',e04/'summary.json',e04/'completion.json',e04/'shared-appearance.pt',root/'datasets/bouncingballs/transforms_train.json',root/'datasets/bouncingballs/transforms_test.json']:
        inputs[path]=digest(path)
    for name,h in control_training['checkpoint_hashes'].items():
        p=e16/name
        assert digest(p)==h
        inputs[p]=h
    for j in range(20):
        p=base/f'test/ours_20000/gt/{j:05d}.png';inputs[p]=digest(p)
    for p,h in inputs.items():assert p.is_relative_to(root) and digest(p)==h,str(p)
    assert json.loads((e04/'completion.json').read_text())['status']=='completed'
    full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy');geometry=np.load(prior/'shared-states.npy',mmap_mode='r')
    grouping=np.load(prior/'shared-grouping.npz');labels=grouping['labels'];reps=grouping['representatives']
    assert full.shape==(150,26936,12) and geometry.shape==(150,k,12) and len(np.unique(labels))==k
    weights=np.load(cache/'opacity.npy').astype(np.float64).reshape(-1);weights/=weights.sum()
    fit=old_manifest['fit_indices'];held=old_manifest['holdout_indices'];assert len(fit)==76 and len(held)==74 and not set(fit)&set(held)
    orders={str(seed):[] for seed in protocol['order_seeds']}
    for seed in protocol['order_seeds']:
        rng=np.random.default_rng(seed);orders[str(seed)]=[rng.permutation(fit).tolist() for _ in range(protocol['epochs'])]
    assert orders['42']==old_manifest['orders']
    assert len({tuple(o) for v in orders.values() for o in v})==60
    out.mkdir(parents=True)
    dump(out/'manifest.json',{'protocol':protocol,'fit':fit,'holdout':held,'orders':orders,'inputs':{str(p):h for p,h in inputs.items()},'GPU':os.environ.get('CUDA_VISIBLE_DEVICES'),'new_gradient_updates':6*1520,'perceptual_beta':protocol['perceptual_beta']})
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['run_perceptual_fit_loss.py','covariance_moment_core.py','test_covariance_moment_core.py','appearance_model.py','reduction_scene.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:
            z.write(repo/'scripts'/name,'scripts/'+name)
        z.write(protocol_path,'experiments/'+protocol_path.name)
    started=time.monotonic();scatter=np.empty((150,k,3,3),dtype=np.float64);meanerr=0.;coverr=0.;momenterr=0.;worst_relative_eigenvalue=0.
    for t in range(150):
        mu,cov,s,mass=group_moments(full[t],labels,weights,k);scatter[t]=s
        meanerr=max(meanerr,float(np.max(abs(mu-geometry[t,:,:3]))));coverr=max(coverr,float(np.max(abs(cov-geometry[t,:,3:].reshape(k,3,3)))))
        for g in [0,k//2,k-1]:
            take=labels==g;p=weights[take]/mass[g];x=np.asarray(full[t,take,:3],dtype=np.float64)
            direct=np.einsum('n,nij->ij',p,np.asarray(full[t,take,3:],dtype=np.float64).reshape(-1,3,3))+np.einsum('n,ni,nj->ij',p,x,x)
            reconstructed=cov[g]+s[g]+np.outer(mu[g],mu[g])
            momenterr=max(momenterr,float(np.max(abs(direct-reconstructed))))
        for c in [cov,cov+s]:
            eig=np.linalg.eigvalsh(c);relative=eig[:,0]/np.maximum(eig[:,-1],1e-30)
            worst_relative_eigenvalue=min(worst_relative_eigenvalue,float(relative.min()))
        if t%30==0:print('MOMENTS',t,flush=True)
    assert meanerr<1e-6 and coverr<1e-7 and momenterr<1e-12
    assert worst_relative_eigenvalue>=-2*np.finfo(np.float32).eps
    features={name:add_scatter(geometry,scatter,coef) for name,coef in protocol['conditions'].items()}
    assert np.array_equal(features['mean_covariance'],geometry)
    assert all(np.array_equal(g[:,:,:3],geometry[:,:,:3]) for g in features.values())
    np.save(out/'center-scatter.npy',scatter.astype(np.float32))
    dump(out/'geometry-audit.json',{'source_group_mean_max_abs_error':meanerr,'source_group_mean_covariance_max_abs_error':coverr,'direct_second_moment_max_abs_error':momenterr,'worst_relative_eigenvalue':worst_relative_eigenvalue,'positions_and_labels_and_counts_unchanged':True,'mean_arm_exact_E02_states':True})
    del scatter
    cfg,pc,scene,pipe=load_scene(base)
    for item in vars(pc).values():
        if isinstance(item,torch.nn.Parameter):item.requires_grad_(False)
        elif isinstance(item,torch.nn.Module):item.requires_grad_(False);item.eval()
    def tensor(x):return torch.tensor(np.array(x,copy=True),device='cuda')
    sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));logits=pc._opacity.detach()
    assert torch.equal(pc.get_features,sh) and torch.equal(pc.get_opacity,opacity)
    ids=tensor(reps).long();frames=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())['frames'];train=scene.getTrainCameras();cameras={}
    for j,r in enumerate(frames):
        i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7 and abs(float(train[j].time)-r['time'])<1e-7
        cameras[i]=small_camera(train[j],protocol['training_size'])
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    def loss_fn(p,t):return .8*(p-t).abs().mean()+.2*(1-ssim(p.unsqueeze(0),t.unsqueeze(0)))
    import lpips
    criterion=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    beta=protocol['perceptual_beta'];assert beta==0.1
    teacher={}
    with torch.no_grad():
        for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
    models={};records=[];hashes={};initial=Appearance(sh,logits,k)
    for seed in protocol['order_seeds']:
        for condition in protocol['conditions']:
            assert shutil.disk_usage(root).free>3*1024**3
            name=f'{condition}__seed{seed}';model=Appearance(sh,logits,k);opt=model.optimizer(protocol['learning_rates'])
            assert all(torch.count_nonzero(getattr(model,n))==0 for n in ['dc','rest','opacity'])
            for epoch,order in enumerate(orders[str(seed)]):
                losses=[]
                for i in order:
                    opt.zero_grad(set_to_none=True);colors,alpha=model(ids)
                    pred=render_state(cameras[i],tensor(features[condition][i]),colors,alpha,cfg.sh_degree)
                    loss=loss_fn(pred,teacher[i])+beta*criterion(pred.unsqueeze(0),teacher[i].unsqueeze(0),normalize=True).mean();assert torch.isfinite(loss);loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    opt.step();losses.append(float(loss.detach()))
                row={'arm':name,'epoch':epoch+1,'updates':(epoch+1)*76,'mean_fit_loss':float(np.mean(losses)),'seconds':time.monotonic()-started};records.append(row)
                (out/'progress.json').write_text(json.dumps(row,indent=2))
                if (epoch+1)%5==0:print('TRAIN',json.dumps(row),flush=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert all(int(state['step'])==1520 for state in opt.state.values())
            path=out/(name+'-appearance.pt');torch.save({'state_dict':{n:v.detach().cpu() for n,v in model.state_dict().items()},'optimizer':opt.state_dict(),'steps':1520,'condition':condition,'seed':seed},path)
            hashes[path.name]=digest(path);models[name]=(condition,model)
    dump(out/'training.json',records)
    dump(out/'training-complete.json',{'all6_frozen_before_evaluation':True,'checkpoint_hashes':hashes,'gradient_updates':9120})
    print('ALL SIX FROZEN; EVALUATING',flush=True)
    teacher.clear();gtrows=[];fidelity=[];temporal=[];previous=None
    def render(condition,i,cam,model):
        colors,alpha=model(ids)
        return render_state(cam,tensor(features[condition][i]),colors,alpha,cfg.sh_degree)
    def iter_models():
        for c in protocol['conditions']:yield c+'__before',c,initial
        for n,(c,m) in models.items():yield n,c,m
    names=[n for n,c,m in iter_models()]
    with torch.no_grad():
        tests=json.loads((root/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
        for j,r in enumerate(tests):
            i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7
            gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
            for name,c,m in iter_models():
                pred=(render(c,i,scene.getTestCameras()[j],m)*255).round()/255;p=pred.unsqueeze(0)
                gtrows.append({'arm':name,'view':j,'PSNR':float(psnr(p,gt)),'SSIM':float(ssim(p,gt)),'LPIPS_alex_v01':float(criterion(p,gt,normalize=True))})
        del criterion
        print('GT COMPLETE',flush=True)
        fixed=small_camera(scene.getTestCameras()[10],256)
        for i,t in enumerate(times):
            target=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree)
            for name,c,m in iter_models():
                pred=render(c,i,cameras[i],m)
                fidelity.append({'arm':name,'time_index':i,'split':'fit' if i in fit else 'holdout','PSNR_to_teacher':float(psnr(pred.unsqueeze(0),target.unsqueeze(0))),'L1_to_teacher':float((pred-target).abs().mean()),'mixed_loss_to_teacher':float(loss_fn(pred,target))})
            images={'teacher':render_state(fixed,tensor(full[i]),sh,opacity,cfg.sh_degree)}
            for name,c,m in iter_models():images[name]=render(c,i,fixed,m)
            if previous is not None:
                for name in names:
                    residual=float(((images[name]-previous[name])-(images['teacher']-previous['teacher'])).abs().mean())
                    temporal.append({'arm':name,'time_index':i,'residual':residual})
            if i in [0,75,149]:
                columns=['teacher','mean_covariance__seed42','moment_covariance__seed42'];panel=Image.new('RGB',(3*256,280),'white');draw=ImageDraw.Draw(panel)
                for col,name in enumerate(columns):
                    draw.text((col*256+3,4),name,fill='black');panel.paste(Image.fromarray((images[name].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)),(col*256,24))
                panel.save(out/f'comparison-time{i:03d}.png')
            previous=images
            if i%50==0:print('FIDELITY/TEMPORAL',i,flush=True)
    dump(out/'gt-per-view.json',gtrows);dump(out/'fidelity-per-view.json',fidelity);dump(out/'temporal-per-pair.json',temporal)
    rows=[]
    for name in names:
        gt=[r for r in gtrows if r['arm']==name];tmp=[r['residual'] for r in temporal if r['arm']==name]
        row={'arm':name,**{metric:float(np.mean([r[metric] for r in gt])) for metric in ['PSNR','SSIM','LPIPS_alex_v01']},'temporal_residual256':float(np.mean(tmp))}
        for split in ['fit','holdout']:
            vals=[r for r in fidelity if r['arm']==name and r['split']==split]
            row[split]={metric:float(np.mean([r[metric] for r in vals])) for metric in ['PSNR_to_teacher','L1_to_teacher','mixed_loss_to_teacher']}
        rows.append(row)
    byname={r['arm']:r for r in rows};paired=[]
    for seed in protocol['order_seeds']:
        a=byname[f'mean_covariance__seed{seed}'];b=byname[f'moment_covariance__seed{seed}']
        paired.append({'seed':seed,**{metric:b[metric]-a[metric] for metric in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']},'holdout_mixed_loss_delta':b['holdout']['mixed_loss_to_teacher']-a['holdout']['mixed_loss_to_teacher']})
    old=json.loads((e04/'summary.json').read_text())
    beforediff=byname['mean_covariance__before']['PSNR']-old['official_20_view_gt']['shared_before']['PSNR']
    afterdiff=byname['mean_covariance__seed42']['PSNR']-old['official_20_view_gt']['shared_after']['PSNR']
    assert abs(beforediff)<1e-4,beforediff
    control_byname={r['arm']:r for r in controls['rows']}
    beta_pairs=[]
    for seed in protocol['order_seeds']:
        for condition in protocol['conditions']:
            name=f'{condition}__seed{seed}'
            new=byname[name];old_control=control_byname[name]
            beta_pairs.append({'seed':seed,'condition':condition,**{metric:new[metric]-old_control[metric] for metric in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256']},'holdout_mixed_loss_delta':new['holdout']['mixed_loss_to_teacher']-old_control['holdout']['mixed_loss_to_teacher']})
    beta_mean={condition:{metric:float(np.mean([r[metric] for r in beta_pairs if r['condition']==condition])) for metric in ['PSNR','SSIM','LPIPS_alex_v01','temporal_residual256','holdout_mixed_loss_delta']} for condition in protocol['conditions']}
    report={'rows':rows,'control_rows':controls['rows'],'paired_beta_minus_E16':beta_pairs,'paired_beta_mean':beta_mean,'paired_moment_minus_mean':paired,'paired_mean':{m:float(np.mean([r[m] for r in paired])) for m in paired[0] if m!='seed'},'E04_initial_reproduction_PSNR_delta':beforediff,'beta_seed42_minus_E04_PSNR':afterdiff,'scope':protocol['scope'],'verification':{'all6_frozen_before_evaluation':True,'E16_controls_hash_verified':True,'same_initial_appearance_and_orders':True,'geometry_audit_passed':True,'same_1520_steps':True,'E04_initial_reproduced':True}}
    dump(out/'report.json',report)
    for p,h in inputs.items():assert digest(p)==h,str(p)
    for n,h in hashes.items():assert digest(out/n)==h,n
    dump(out/'completion.json',{'status':'completed','E16_controls_hash_verified':True,'inputs_unchanged':True,'checkpoint_hashes_verified':True,'source_sha256':digest(out/'source.zip'),'new_gradient_updates':9120,'seconds':time.monotonic()-started})
    print('RESULT',json.dumps(report),flush=True)
if __name__=='__main__':main()
