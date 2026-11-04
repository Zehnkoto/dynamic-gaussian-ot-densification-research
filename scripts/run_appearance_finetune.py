#!/usr/bin/env python3
"""E04 fixed-final-step paired appearance distillation; immutable inputs."""
import argparse,json,os,pathlib,time,zipfile
import numpy as np
import torch
from appearance_model import Appearance
from reduction_scene import load_scene,small_camera,render_state
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--mode',choices=['joint','sh','opacity'],default='joint');ap.add_argument('--seed',type=int);ap.add_argument('--protocol',default='appearance-finetune-v1.json');args=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=args.output.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('Unsafe or existing output')
    protocol=json.loads((repo/'experiments'/args.protocol).read_text());k=protocol['budget'];protocol['mode']=args.mode
    if args.seed is not None:protocol['seed']=args.seed
    base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z'
    cache=root/'outputs/ot/e02-ctd-20261103T082303Z/trajectory-cache';prior=root/'outputs/ot/e02-ctd-refined-20261103T082856Z'/f'budget-{k}'
    split=json.loads((repo/'experiments/reduction-ctd-v2.json').read_text());meta=json.loads((cache/'manifest.json').read_text());validation=json.loads((base/'baseline-validation.json').read_text())
    assert meta['baseline_model_hashes']==validation['model_files']
    inputs={cache/n:r['sha256'] for n,r in meta['files'].items()};inputs.update({base/n:r['sha256'] for n,r in validation['model_files'].items()})
    for n in ['shared-states.npy','perframe-states.npy','shared-grouping.npz','perframe-representatives.npy']:inputs[prior/n]=digest(prior/n)
    for p,h in inputs.items():assert digest(p)==h,str(p)
    cfg,pc,scene,pipe=load_scene(base)
    for a in vars(pc).values():
        if isinstance(a,torch.nn.Parameter):a.requires_grad_(False)
        elif isinstance(a,torch.nn.Module):a.requires_grad_(False);a.eval()
    def tensor(a):return torch.tensor(np.array(a,copy=True),device='cuda')
    sh=tensor(np.load(cache/'sh.npy'));opacity=tensor(np.load(cache/'opacity.npy'));logits=pc._opacity.detach()
    assert torch.equal(pc.get_features,sh) and torch.equal(torch.sigmoid(logits),opacity)
    times=np.load(cache/'times.npy');full=np.load(cache/'states.npy',mmap_mode='r')
    features={m:np.load(prior/(m+'-states.npy'),mmap_mode='r') for m in ['shared','perframe']}
    reps={'shared':np.load(prior/'shared-grouping.npz')['representatives'],'perframe':np.load(prior/'perframe-representatives.npy')}
    def ids(m,i):return tensor(reps[m] if m=='shared' else reps[m][i]).long()
    def index(t):
        i=int(np.argmin(abs(times-t)));assert abs(times[i]-t)<1e-7;return i
    fit=[index(r['time']) for r in split['fit']];held=[index(r['time']) for r in split['reduction_holdout']]
    assert len(fit)==76 and len(held)==74 and not set(fit)&set(held)
    frames=json.loads((root/'datasets/bouncingballs/transforms_train.json').read_text())['frames'];train=scene.getTrainCameras();cameras={}
    for j,r in enumerate(frames):
        cam=train[j];assert abs(float(cam.time)-r['time'])<1e-7;cameras[index(r['time'])]=small_camera(cam,protocol['training_size'])
    assert len(cameras)==150
    rng=np.random.default_rng(protocol['seed']);orders=[rng.permutation(fit).tolist() for _ in range(protocol['epochs'])]
    out.mkdir(parents=True);start=time.monotonic()
    dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in inputs.items()},'fit_indices':fit,'holdout_indices':held,'orders':orders,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES'),'trainable_parameters_per_arm':k*({'joint':49,'sh':48,'opacity':1}[args.mode])})
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in ['run_appearance_finetune.py','appearance_model.py','evaluate_appearance.py','reduction_scene.py','run_ctd_comparison.py','reduction_core.py','inspect_baseline.py','common.sh']:z.write(repo/'scripts'/n,'scripts/'+n)
        for n in [args.protocol,'reduction-ctd-v2.json']:z.write(repo/'experiments'/n,'experiments/'+n)
    from utils.loss_utils import ssim
    def render(m,i,cam,appearance=None):
        selected=ids(m,i);a,o=appearance(selected) if appearance is not None else (sh[selected],opacity[selected])
        return render_state(cam,tensor(features[m][i]),a,o,cfg.sh_degree)
    teacher={}
    with torch.no_grad():
        for i in fit:teacher[i]=render_state(cameras[i],tensor(full[i]),sh,opacity,cfg.sh_degree).detach()
    models={m:Appearance(sh,logits,k,args.mode) for m in features};checks={}
    def loss_fn(pred,target):return .8*(pred-target).abs().mean()+.2*(1-ssim(pred.unsqueeze(0),target.unsqueeze(0)))
    for m,model in models.items():
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==k*({'joint':49,'sh':48,'opacity':1}[args.mode])
        pred=render(m,fit[0],cameras[fit[0]],model)
        with torch.no_grad():err=float((pred-render(m,fit[0],cameras[fit[0]])).abs().max())
        assert err<1e-7
        assert all(torch.count_nonzero(p)==0 for p in model.parameters() if not p.requires_grad)
        loss_fn(pred,teacher[fit[0]]).backward();grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.requires_grad}
        assert all(np.isfinite(v) and v>0 for v in grads.values())
        assert all(not p.requires_grad for p in [pc._xyz,pc._scaling,pc._rotation,pc._opacity])
        checks[m]={'zero_residual_render_max_error':err,'gradient_L1':grads};model.zero_grad(set_to_none=True)
    dump(out/'preflight.json',checks);print('PREFLIGHT PASSED',json.dumps(checks),flush=True)
    log=[]
    for m,model in models.items():
        optimizer=model.optimizer(protocol['learning_rates'])
        for epoch,order in enumerate(orders):
            losses=[]
            for i in order:
                optimizer.zero_grad(set_to_none=True);loss=loss_fn(render(m,i,cameras[i],model),teacher[i]);assert torch.isfinite(loss)
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            row={'arm':m,'epoch':epoch+1,'updates':(epoch+1)*len(fit),'mean_loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-start};log.append(row);print('TRAIN',json.dumps(row),flush=True)
            with (out/'progress.json').open('w') as f:json.dump(row,f,indent=2)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(torch.count_nonzero(p)==0 for p in model.parameters() if not p.requires_grad)
        torch.save({'state_dict':{n:t.detach().cpu() for n,t in model.state_dict().items()},'optimizer':optimizer.state_dict(),'protocol':protocol,'steps':protocol['epochs']*76},out/(m+'-appearance.pt'))
    dump(out/'training.json',log);teacher.clear();print('TRAINING COMPLETE; evaluating fixed final weights',flush=True)
    from evaluate_appearance import evaluate
    evaluate(out,base,prior,protocol,times,fit,full,sh,opacity,cfg,scene,cameras,models,render,tensor)
    for p,h in inputs.items():assert digest(p)==h,'Input modified: '+str(p)
    dump(out/'completion.json',{'status':'completed','source_sha256':digest(out/'source.zip'),'input_hashes_unchanged':True,'elapsed_seconds':time.monotonic()-start});print('COMPLETE',str(out),flush=True)
if __name__=='__main__':main()
