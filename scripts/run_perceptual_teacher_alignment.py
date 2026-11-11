#!/usr/bin/env python3
"""E17b: compare both frozen E16 students to GT and frozen full teacher."""
import argparse,json,os,pathlib,zipfile,shutil
import numpy as np
import torch
from PIL import Image
from appearance_model import Appearance
from covariance_moment_core import add_scatter
from reduction_scene import load_scene,render_state
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research';out=a.output.resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('Unsafe or existing output')
    assert shutil.disk_usage(root).free>3*1024**3
    proto=repo/'experiments/perceptual-teacher-alignment-v1.json';protocol=json.loads(proto.read_text())
    e16=root/'outputs/ot/e16-covariance-moment-diagnostic-20261110-v1'
    e17=root/'outputs/ot/e17-perceptual-diagnostic-20261110-v1'
    cache=root/'outputs/ot/e02-ctd-20261103T082303Z/trajectory-cache'
    prior=root/'outputs/ot/e02-ctd-refined-20261103T082856Z/budget-10000'
    base=root/'outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z'
    ckpts={arm:e16/f'{arm}_covariance__seed42-appearance.pt' for arm in ['mean','moment']}
    files=[proto,e16/'audit.json',e16/'completion.json',e17/'report.json',e17/'per-view.json',cache/'states.npy',cache/'sh.npy',cache/'opacity.npy',cache/'times.npy',prior/'shared-states.npy',prior/'shared-grouping.npz',e16/'center-scatter.npy',root/'datasets/bouncingballs/transforms_test.json',*ckpts.values()]
    files += [base/f'test/ours_20000/gt/{j:05d}.png' for j in range(20)]
    hashes={p:digest(p) for p in files};assert json.loads((e16/'completion.json').read_text())['status']=='completed'
    out.mkdir()
    dump(out/'manifest.json',{'protocol':protocol,'inputs':{str(p):h for p,h in hashes.items()},'GPU':os.environ.get('CUDA_VISIBLE_DEVICES')})
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['run_perceptual_teacher_alignment.py','appearance_model.py','covariance_moment_core.py','reduction_scene.py','inspect_baseline.py','run_ctd_comparison.py','common.sh']:
            z.write(repo/'scripts'/name,'scripts/'+name)
        z.write(proto,'experiments/'+proto.name)
    cfg,pc,scene,pipe=load_scene(base)
    for item in vars(pc).values():
        if isinstance(item,torch.nn.Parameter):item.requires_grad_(False)
        elif isinstance(item,torch.nn.Module):item.requires_grad_(False);item.eval()
    def tensor(a):return torch.tensor(np.array(a,copy=True),device='cuda')
    sh=tensor(np.load(cache/'sh.npy'));op=tensor(np.load(cache/'opacity.npy'))
    assert torch.equal(sh,pc.get_features) and torch.equal(op,pc.get_opacity)
    full=np.load(cache/'states.npy',mmap_mode='r');times=np.load(cache/'times.npy')
    geo=np.load(prior/'shared-states.npy',mmap_mode='r');scatter=np.load(e16/'center-scatter.npy',mmap_mode='r')
    ids=tensor(np.load(prior/'shared-grouping.npz')['representatives']).long()
    models={}
    for arm,path in ckpts.items():
        c=torch.load(path,map_location='cuda');model=Appearance(sh,pc._opacity.detach(),10000);model.load_state_dict(c['state_dict']);model.eval();models[arm]=model
    import lpips
    metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval()
    test=json.loads((root/'datasets/bouncingballs/transforms_test.json').read_text())['frames']
    e17rows={(r['view'],r['arm']):r for r in json.loads((e17/'per-view.json').read_text())}
    rows=[]
    with torch.no_grad():
        for j,r in enumerate(test):
            i=int(np.argmin(abs(times-r['time'])));assert abs(times[i]-r['time'])<1e-7
            cam=scene.getTestCameras()[j]
            gt=tensor(np.asarray(Image.open(base/f'test/ours_20000/gt/{j:05d}.png').convert('RGB'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
            teacher=(render_state(cam,tensor(full[i]),sh,op,cfg.sh_degree)*255).round()/255
            teacher=teacher.unsqueeze(0)
            for arm,model in models.items():
                state=geo[i] if arm=='mean' else add_scatter(geo[i],scatter[i],1)
                colors,alpha=model(ids);p=(render_state(cam,tensor(state),colors,alpha,cfg.sh_degree)*255).round()/255;p=p.unsqueeze(0)
                row={'view':j,'arm':arm,'LPIPS_to_GT':float(metric(p,gt,normalize=True)),
                     'LPIPS_to_teacher':float(metric(p,teacher,normalize=True)),
                     'MSE_to_teacher':float(((p-teacher)**2).mean()),'L1_to_teacher':float((p-teacher).abs().mean())}
                assert abs(row['LPIPS_to_GT']-e17rows[(j,arm)]['LPIPS'])<2e-4
                rows.append(row)
            print('VIEW',j,rows[-2]['LPIPS_to_teacher'],rows[-1]['LPIPS_to_teacher'],flush=True)
    dump(out/'per-view.json',rows)
    paired=[];by={(r['view'],r['arm']):r for r in rows}
    for j in range(20):
        a=by[j,'mean'];b=by[j,'moment']
        paired.append({'view':j,**{k+'_moment_minus_mean':b[k]-a[k] for k in ['LPIPS_to_GT','LPIPS_to_teacher','MSE_to_teacher','L1_to_teacher']}})
    dump(out/'paired.json',paired)
    result={'mean':{arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in ['LPIPS_to_GT','LPIPS_to_teacher','MSE_to_teacher','L1_to_teacher']} for arm in ['mean','moment']},
            'positive_delta_views':{k:sum(r[k+'_moment_minus_mean']>0 for r in paired) for k in ['LPIPS_to_GT','LPIPS_to_teacher','MSE_to_teacher','L1_to_teacher']},
            'interpretation_boundaries':protocol['limits']}
    dump(out/'report.json',result)
    for p,h in hashes.items():assert digest(p)==h,p
    dump(out/'completion.json',{'status':'completed','no_training':True,'source_sha256':digest(out/'source.zip'),'inputs_unchanged':True})
    print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
