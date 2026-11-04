#!/usr/bin/env python3
"""Bounded sequential split robustness matrix; no evaluation-driven selection."""
import argparse,json,os,pathlib,shutil,subprocess,sys,time,zipfile
from run_ctd_comparison import dump,digest
ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--protocol',default='split-robustness-v2.json');a=ap.parse_args();root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=a.output.resolve();repo=root/'projects/dynamic-gaussian-ot-densification-research'
assert out.is_relative_to(root) and not out.exists();protocol=json.loads((repo/'experiments'/a.protocol).read_text());old=root/'outputs/ot/e05-split-20261104T103723Z';assert json.loads((old/'validation/completion.json').read_text())['status']=='completed'
out.mkdir(parents=True);records=[];start=time.monotonic()
with zipfile.ZipFile(out/'batch-source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in ['run_split_robustness.py','prepare_split_proposals.py','run_split_validation.py','evaluate_split_validation.py','summarize_split_validation.py','split_proposal_core.py']:z.write(repo/'scripts'/name,'scripts/'+name)
 z.write(repo/'experiments'/a.protocol,'experiments/'+a.protocol)
dump(out/'manifest.json',{'protocol':protocol,'GPU':os.environ.get('CUDA_VISIBLE_DEVICES'),'batch_source_sha256':digest(out/'batch-source.zip')})
for condition in protocol['conditions']:
 budget=condition['splits'];seed=condition['seed'];reuse=protocol.get('reuse_runs',{}).get(str(budget)+':'+str(seed));dest=(root/reuse).resolve() if reuse else old if budget==256 and seed==42 else out/f'splits{budget}-seed{seed}'
 assert dest.is_relative_to(root)
 if dest==old or reuse:
  assert json.loads((dest/'validation/completion.json').read_text())['status']=='completed'
  records.append(dict(condition,output=str(dest),reused=True));continue
 if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('Less than5GiB remaining on shared storage; stop before next condition')
 commands=[['scripts/prepare_split_proposals.py','--output',str(dest),'--protocol',a.protocol,'--splits',str(budget),'--seed',str(seed)],['scripts/run_split_validation.py','--input',str(dest)],['scripts/summarize_split_validation.py',str(dest)]]
 with (out/f'splits{budget}-seed{seed}.log').open('x') as log:
  for command in commands:
   print('START',budget,seed,command[0],flush=True);r=subprocess.run([sys.executable,'-u']+command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
   if r.returncode:
    dump(out/'failure.json',{'condition':condition,'stage':command[0],'exit_code':r.returncode});raise SystemExit(r.returncode)
 records.append(dict(condition,output=str(dest),reused=False))
 with (out/'progress.json').open('w') as f:json.dump(records,f,indent=2)
 print('DONE',budget,seed,'conditions',len(records),'/5',flush=True)
dump(out/'runs.json',records);dump(out/'completion.json',{'status':'completed','conditions':len(records),'elapsed_seconds':time.monotonic()-start});print('ROBUSTNESS COMPLETE',flush=True)
