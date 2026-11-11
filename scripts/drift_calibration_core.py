"""Fit-only log-bracket calibration; no rendering or evaluation metrics accepted."""
import math

def matched(drift,target,tolerance):
    return abs(drift/target-1)<=tolerance

def choose_candidate(records,target,tolerance):
    passing=[r for r in records if matched(r['fit_drift'],target,tolerance)]
    if passing:return passing[0]
    return min(records,key=lambda r:(abs(math.log(max(r['fit_drift'],1e-30)/target)),r['lambda']))

def next_lambda(records,target):
    above=[r for r in records if r['fit_drift']>target]
    below=[r for r in records if r['fit_drift']<target]
    pairs=[(a,b) for a in above for b in below if a['lambda']<b['lambda']]
    if not pairs:raise ValueError('No positive monotone fit-only bracket')
    a,b=min(pairs,key=lambda ab:math.log(ab[1]['lambda']/ab[0]['lambda']))
    fraction=math.log(a['fit_drift']/target)/math.log(a['fit_drift']/b['fit_drift'])
    fraction=max(.1,min(.9,fraction))
    value=math.exp(math.log(a['lambda'])+fraction*math.log(b['lambda']/a['lambda']))
    if any(abs(math.log(value/r['lambda']))<1e-10 for r in records):raise ValueError('Repeated calibration lambda')
    return value
