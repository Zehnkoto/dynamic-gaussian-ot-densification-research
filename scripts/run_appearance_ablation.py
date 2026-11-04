#!/usr/bin/env python3
"""Sequential predeclared 3-mode x 3-view-order-seed sweep, reusing v1 joint42."""
import argparse,json,os,pathlib,subprocess,sys,time,zipfile
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=args.output.resolve()
if not out.is_relative_to(root) or out.exists():raise ValueError('Unsafe or existing output')
repo=root/'projects/dynamic-gaussian-ot-densification-research';protocol=json.loads((repo/'experiments/appearance-ablation-v2.json').read_text());reused=root/'outputs/ot/e04-appearance-20261103T095355Z'
assert json.loads((reused/'completion.json').read_text())['status']=='completed'
out.mkdir(parents=True)
with zipfile.ZipFile(out/'batch-source.zip','x') as z:
 for n in ['run_appearance_ablation.py','run_appearance_finetune.py','appearance_model.py','evaluate_appearance.py']:z.write(repo/'scripts'/n,'scripts/'+n)
 z.write(repo/'experiments/appearance-ablation-v2.json','experiments/appearance-ablation-v2.json')
records=[{'mode':'joint','seed':42,'output':str(reused),'reused':True}];start=time.monotonic()
dump(out/'manifest.json',{'protocol':protocol,'reused_run':str(reused),'GPU':os.environ.get('CUDA_VISIBLE_DEVICES'),'batch_source_sha256':digest(out/'batch-source.zip')})
for seed in protocol['seeds']:
 for mode in protocol['modes']:
  if seed==42 and mode=='joint':continue
  dest=out/f'{mode}-seed{seed}';command=[sys.executable,'-u',str(repo/'scripts/run_appearance_finetune.py'),'--output',str(dest),'--seed',str(seed),'--mode',mode,'--protocol','appearance-ablation-v2.json']
  print('START',mode,seed,flush=True)
  with (out/f'{mode}-seed{seed}.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=repo)
  if r.returncode:
   dump(out/'failure.json',{'mode':mode,'seed':seed,'exit_code':r.returncode});raise SystemExit(r.returncode)
  assert json.loads((dest/'completion.json').read_text())['status']=='completed'
  records.append({'mode':mode,'seed':seed,'output':str(dest),'reused':False});print('DONE',mode,seed,'conditions',len(records),'/9',flush=True)
  with (out/'progress.json').open('w') as f:json.dump(records,f,indent=2)
dump(out/'runs.json',records);dump(out/'completion.json',{'status':'completed','conditions':len(records),'elapsed_seconds':time.monotonic()-start});print('BATCH COMPLETE',out,flush=True)
