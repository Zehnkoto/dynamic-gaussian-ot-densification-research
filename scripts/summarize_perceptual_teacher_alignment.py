#!/usr/bin/env python3
"""Audit frozen teacher/GT perceptual diagnostic against E16 and E17."""
import argparse,json,os,pathlib,zipfile
import numpy as np
from run_ctd_comparison import digest,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=pathlib.Path);a=ap.parse_args()
    root=pathlib.Path(os.environ['RESEARCH_ROOT']).resolve();out=a.run.resolve();assert out.is_relative_to(root)
    read=lambda n:json.loads((out/n).read_text())
    manifest=read('manifest.json');completion=read('completion.json');rows=read('per-view.json');paired=read('paired.json');report=read('report.json')
    assert completion['status']=='completed' and completion['no_training'] and completion['inputs_unchanged']
    assert digest(out/'source.zip')==completion['source_sha256']
    for p,h in manifest['inputs'].items():assert digest(pathlib.Path(p))==h,p
    assert len(rows)==40 and len(paired)==20
    seen={(r['view'],r['arm']):r for r in rows};assert len(seen)==40
    assert all((j,arm) in seen for j in range(20) for arm in ['mean','moment'])
    e17=root/'outputs/ot/e17-perceptual-diagnostic-20261110-v1'
    old={(r['view'],r['arm']):r for r in json.loads((e17/'per-view.json').read_text())}
    maxerr=0.
    for j in range(20):
        for arm in ['mean','moment']:
            r=seen[j,arm];maxerr=max(maxerr,abs(r['LPIPS_to_GT']-old[j,arm]['LPIPS']))
        a=seen[j,'mean'];b=seen[j,'moment'];q=paired[j];assert q['view']==j
        for k in ['LPIPS_to_GT','LPIPS_to_teacher','MSE_to_teacher','L1_to_teacher']:
            assert abs(q[k+'_moment_minus_mean']-(b[k]-a[k]))<1e-12
    assert maxerr<2e-4
    metrics=['LPIPS_to_GT','LPIPS_to_teacher','MSE_to_teacher','L1_to_teacher']
    for arm in ['mean','moment']:
        for metric in metrics:
            assert abs(report['mean'][arm][metric]-np.mean([seen[j,arm][metric] for j in range(20)]))<1e-12
    counts={k:sum(q[k+'_moment_minus_mean']>0 for q in paired) for k in metrics}
    assert counts==report['positive_delta_views']
    assert counts['LPIPS_to_GT']==counts['LPIPS_to_teacher']==20
    assert counts['MSE_to_teacher']==0
    assert counts['L1_to_teacher']==1
    summary={'verification':{'all_input_and_source_hashes_match':True,'forty_rows_and_twenty_pairs_recomputed':True,'GT_LPIPS_reproduces_E17_max_abs_error':maxerr,
                             'no_training':True},
             'paired_mean_deltas':{k:float(np.mean([q[k+'_moment_minus_mean'] for q in paired])) for k in metrics},
             'positive_delta_views':counts,
             'interpretation':'On the same20 views, moment matching decreases pixel MSE to teacher in every view but increases LPIPS to teacher in every view. Teacher-vs-GT bias alone cannot explain the LPIPS tradeoff; teacher-directed pixel loss and perceptual similarity are misaligned here.'}
    dump(out/'audit.json',summary)
    repo=root/'projects/dynamic-gaussian-ot-densification-research'
    with zipfile.ZipFile(out/'analysis-source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:z.write(repo/'scripts/summarize_perceptual_teacher_alignment.py','scripts/summarize_perceptual_teacher_alignment.py')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
