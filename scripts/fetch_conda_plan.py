#!/usr/bin/env python3
"""Fetch an official Conda dry-run plan sequentially with resume and SHA256 checks."""
import hashlib, json, os, pathlib, subprocess, sys
root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
plan=pathlib.Path(sys.argv[1]).resolve()
if not plan.is_relative_to(root): raise SystemExit('Plan must be inside workspace')
cache=root/'cache/conda/pkgs'
for item in json.loads(plan.read_text()).get('actions',{}).get('FETCH',[]):
    name=item['fn']; url=item['url']; expected=item['sha256']
    if pathlib.Path(name).name!=name or not url.startswith('https://conda.anaconda.org/conda-forge/'):
        raise ValueError('Unexpected package source')
    target=cache/name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected: continue
    part=cache/(name+'.resume')
    for attempt in range(4):
        print('Fetch',name,'attempt',attempt+1,flush=True)
        result=subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15',
                               '--max-time','180','--continue-at','-','-o',str(part),url])
        if part.exists() and hashlib.sha256(part.read_bytes()).hexdigest()==expected:
            part.replace(target); break
        if result.returncode==0: raise ValueError('Package hash mismatch: '+name)
    else: raise SystemExit('Could not fetch '+name+'; partial download preserved')
explicit=['@EXPLICIT']
for item in json.loads(plan.read_text()).get('actions',{}).get('LINK',[]):
    name=item['dist_name']
    candidates=[cache/(name+'.conda'),cache/(name+'.tar.bz2')]
    artifact=next((p for p in candidates if p.is_file()),None)
    if artifact is None: raise FileNotFoundError('Missing local artifact: '+name)
    if not artifact.resolve().is_relative_to(root): raise ValueError('Unsafe local artifact')
    explicit.append(artifact.as_uri()+'#'+hashlib.md5(artifact.read_bytes()).hexdigest())
plan.with_suffix('.explicit.txt').write_text('\n'.join(explicit)+'\n')
print('All planned artifacts verified; local explicit install list written')
