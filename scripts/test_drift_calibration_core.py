import math
from drift_calibration_core import matched,choose_candidate,next_lambda
r=[{'lambda':.001,'fit_drift':.1},{'lambda':.1,'fit_drift':.001}]
v=next_lambda(r,.01);assert math.isclose(v,.01)
assert not matched(.0104,.01,.03) and matched(.0102,.01,.03)
r+= [{'lambda':.009,'fit_drift':.0102},{'lambda':.01,'fit_drift':.0101}]
assert choose_candidate(r,.01,.03) is r[2] # first pass, not GT/closest selected
try:next_lambda([{'lambda':.1,'fit_drift':.2}],.01)
except ValueError:pass
else:raise AssertionError('Must have an actual bracket')
print('Fit-only interpolation, first-pass stop, tolerance and missing-bracket checks passed')
