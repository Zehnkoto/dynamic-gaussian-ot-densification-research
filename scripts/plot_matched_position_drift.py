"""Qualified E15 standalone plot; retain the original audit figure unchanged."""
import argparse,json,pathlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('output',type=pathlib.Path);out=ap.parse_args().output.resolve();r=json.loads((out/'analysis/report.json').read_text());dest=out/'analysis/matched-position-drift-qualified.png';assert not dest.exists()
fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
for ax,key,title in zip(axes,['PSNR','fit_CTD_relative_increase','LPIPS_alex_v01'],['GT PSNR (higher)','Actual fit CTD increase','GT LPIPS-Alex v0.1 (lower)']):
 for method,label,marker in [('ctd','Source-mass CTD','o'),('l2','Uniform L2','s')]:
  selected=[a for a in r['aggregates'] if a['method']==method and a['scope']=='confirmation_orders'];ax.errorbar([a['budget']*100 for a in selected],[a['metrics'][key]['mean'] for a in selected],yerr=[a['metrics'][key]['sample_std'] for a in selected],fmt=marker+'-',capsize=3,label=label)
 if key=='fit_CTD_relative_increase':ax.plot([1,4],[.01,.04],color='gray',ls='--',label='Target')
 ax.set_xticks([1,4],['1%*','4%']);ax.set_xlabel('Predeclared fit CTD drift target');ax.set_title(title);ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('E15: fit-only order42 calibration; frozen lambda on43/44; mean +/- sample SD (n=2 orders)\n*1%: both order43 arms miss predefined +/-3% tolerance. 4%: all arms matched.',fontsize=12)
fig.savefig(dest,dpi=140);plt.close(fig)
source=out/'analysis/qualified-plot-source.py';assert not source.exists();source.write_text(pathlib.Path(__file__).read_text())
print(dest)
