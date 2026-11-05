"""Squared-error decomposition for fixed teacher and real-image targets."""
def mse_decomposition(pred,teacher,gt):
    if pred.shape!=teacher.shape or pred.shape!=gt.shape:
        raise ValueError('Pred, teacher and GT must share shape')
    a=pred-teacher;b=teacher-gt
    return {'teacher_mse':float(a.square().mean()),'teacher_gt_mse':float(b.square().mean()),'cross_term':float(2*(a*b).mean()),'gt_mse':float((pred-gt).square().mean())}
