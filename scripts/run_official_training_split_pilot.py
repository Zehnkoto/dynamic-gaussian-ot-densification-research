#!/usr/bin/env python3
"""E22: genuine official 4DGS trainable split operator, small controlled continuation."""
import argparse
import hashlib
import json
import os
import pathlib
import random
import time

import numpy as np
import torch

from inspect_baseline import configuration
from run_gaussian_w2_scale_audit import bures_covariance, self_test


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    repo = root / 'projects/dynamic-gaussian-ot-densification-research'
    four = root / 'projects/4DGaussians'
    base = root / 'outputs/dnerf/jumpingjacks/baseline-e19-jumpingjacks-20261110-v1'
    e19 = root / 'outputs/ot/e19-second-scene-jumpingjacks-20261110-v1'
    e21 = root / 'outputs/ot/e21-gaussian-w2-scale-20261110-v1'
    data = root / 'datasets/jumpingjacks'
    out = args.output.resolve()
    assert out.is_relative_to(root / 'outputs/ot') and not out.exists()
    import shutil
    assert shutil.disk_usage(root).free > 5 * 1024**3
    protocol_path = repo / 'experiments/official-training-split-pilot-v1.json'
    protocol = json.loads(protocol_path.read_text())
    assert json.loads((e19 / 'audit.json').read_text())['status'] == 'passed'
    assert json.loads((e21 / 'audit.json').read_text())['status'] == 'passed'
    cfg = configuration(base / 'cfg_args')
    assert pathlib.Path(cfg.source_path).resolve() == data and pathlib.Path(cfg.model_path).resolve() == base
    import subprocess
    assert subprocess.check_output(['git', '-C', str(four), 'rev-parse', 'HEAD'], text=True).strip() == '843d5ac636c37e4b611242287754f3d4ed150144'
    assert not subprocess.check_output(['git', '-C', str(four), 'status', '--porcelain'], text=True).strip()
    inputs = [protocol_path, repo / 'scripts/run_official_training_split_pilot.py', repo / 'scripts/run_gaussian_w2_scale_audit.py',
              base / 'cfg_args', *sorted((base / 'point_cloud/iteration_20000').glob('*')),
              e19 / 'manifest.json', e19 / 'shared-grouping.npz', e19 / 'teacher-states.npy', e19 / 'mean-states.npy', e21 / 'audit.json']
    hashes = {str(path): digest(path) for path in inputs}
    self_test()
    import sys
    sys.path.insert(0, str(four))
    from arguments import ModelParams, ModelHiddenParams, PipelineParams, OptimizationParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from utils.image_utils import psnr
    import lpips
    ap = argparse.ArgumentParser()
    mp, hp, pp, op = ModelParams(ap), ModelHiddenParams(ap), PipelineParams(ap), OptimizationParams(ap)
    hyper, pipe, options = hp.extract(cfg), pp.extract(cfg), op.extract(cfg)
    torch.manual_seed(6666)
    np.random.seed(6666)
    random.seed(6666)
    pc = GaussianModel(cfg.sh_degree, hyper)
    scene = Scene(mp.extract(cfg), pc, load_iteration=20000, shuffle=False)
    assert len(pc.get_xyz) == 24985 and len(scene.getTrainCameras()) == 200 and len(scene.getTestCameras()) == 20
    # Scene.load_iteration bypasses create_from_pcd, so restore its training-time spatial scale.
    pc.spatial_lr_scale = float(scene.cameras_extent)
    assert pc.spatial_lr_scale > 0
    pc.training_setup(options)
    background = torch.ones(3, device='cuda') if cfg.white_background else torch.zeros(3, device='cuda')
    rng = np.random.default_rng(protocol['training_order_seed'])
    order = rng.integers(0, 200, size=protocol['warmup_updates'] + protocol['post_updates_each']).tolist()
    cameras = scene.getTrainCameras()
    started = time.monotonic()
    out.mkdir(parents=True)
    save_json(out / 'manifest.json', {'protocol': protocol, 'inputs_sha256': hashes, 'train_camera_order': order,
                                      'initial_gaussians': 24985, 'four_repo_commit': '843d5ac636c37e4b611242287754f3d4ed150144'})

    def train_step(model, step, camera_index, collect_gradient=False):
        model.update_learning_rate(20000 + step)
        assert all(group['lr'] > 0 for group in model.optimizer.param_groups if group['name'] in ('xyz', 'deformation', 'grid'))
        camera = cameras[camera_index]
        result = render(camera, model, pipe, background, stage='fine', cam_type=scene.dataset_type)
        prediction = result['render']
        gt = camera.original_image[:3].cuda()
        loss = l1_loss(prediction, gt)
        if cfg.time_smoothness_weight != 0:
            loss = loss + model.compute_regulation(cfg.time_smoothness_weight, cfg.l1_time_planes, cfg.plane_tv_weight)
        if cfg.lambda_dssim != 0:
            loss = loss + cfg.lambda_dssim * (1 - ssim(prediction.unsqueeze(0), gt.unsqueeze(0)))
        assert torch.isfinite(loss)
        loss.backward()
        if collect_gradient:
            assert result['viewspace_points'].grad is not None
            with torch.no_grad():
                model.add_densification_stats(result['viewspace_points'].grad, result['visibility_filter'])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    warm_losses = []
    for step in range(protocol['warmup_updates']):
        warm_losses.append(train_step(pc, step + 1, order[step], collect_gradient=True))
        if (step + 1) % 32 == 0:
            print(f'WARMUP {step+1}/128', flush=True)
    gradients = (pc.xyz_gradient_accum / pc.denom.clamp_min(1)).flatten().detach().cpu().numpy()
    eligible = (pc.get_scaling.max(dim=1).values > pc.percent_dense * scene.cameras_extent).detach().cpu().numpy()
    assert eligible.sum() >= 64 and np.isfinite(gradients).all()
    score_started = time.monotonic()
    full = np.load(e19 / 'teacher-states.npy', mmap_mode='r')
    fixed = np.load(e19 / 'mean-states.npy', mmap_mode='r')
    groups = np.load(e19 / 'shared-grouping.npz')
    labels = groups['labels'].astype(np.int64)
    fit = json.loads((e19 / 'manifest.json').read_text())['fit_indices']
    weights = pc.get_opacity.detach().flatten().cpu().numpy().astype(np.float64)
    weights /= weights.sum()
    score = np.zeros(24985, dtype=np.float64)
    with torch.no_grad():
        for time_index in fit:
            source = full[time_index]
            center = fixed[time_index]
            for lo in range(0, 24985, 4096):
                hi = min(lo + 4096, 24985)
                x = torch.as_tensor(np.array(source[lo:hi], copy=True), device='cuda', dtype=torch.float64)
                y = torch.as_tensor(np.array(center[labels[lo:hi]], copy=True), device='cuda', dtype=torch.float64)
                pos = (x[:, :3] - y[:, :3]).square().sum(1)
                cov, _ = bures_covariance(x[:, 3:].reshape(-1, 3, 3), y[:, 3:].reshape(-1, 3, 3))
                score[lo:hi] += (pos + cov).cpu().numpy() / len(fit)
    score *= weights
    score_seconds = time.monotonic() - score_started
    available = np.flatnonzero(eligible)
    count = protocol['split_parents']
    gradient_ids = available[np.argsort(-gradients[available], kind='stable')[:count]]
    temporal_ids = available[np.argsort(-score[available], kind='stable')[:count]]
    assert len(np.unique(gradient_ids)) == len(np.unique(temporal_ids)) == count
    selected = {'gradient_top64': gradient_ids.tolist(), 'temporal_w2_top64': temporal_ids.tolist()}
    save_json(out / 'selection.json', {'selected': selected, 'eligible_count': int(eligible.sum()),
                                       'overlap': len(set(gradient_ids) & set(temporal_ids)),
                                       'score_seconds': score_seconds,
                                       'gradient_selected_mean': float(gradients[gradient_ids].mean()),
                                       'temporal_selected_mean_score': float(score[temporal_ids].mean()),
                                       'warning': 'Temporal W2 score is compression-target distortion, not GHAP CTD split gain.'})
    torch.save(pc.capture(), out / 'common-warmup.pt')
    scene.gaussians = None
    del pc
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    rows = []
    for arm in protocol['arms']:
        model = GaussianModel(cfg.sh_degree, hyper)
        model._deformation = model._deformation.to('cuda')
        model.restore(torch.load(out / 'common-warmup.pt', map_location='cuda'), options)
        assert len(model.get_xyz) == 24985
        if arm != 'no_split':
            mask = torch.zeros(24985, device='cuda')
            mask[torch.as_tensor(selected[arm], device='cuda', dtype=torch.long)] = 1
            torch.manual_seed(protocol['split_rng_seed'])
            model.densify_and_split(mask, 0.5, scene.cameras_extent, N=2)
            assert len(model.get_xyz) == 24985 + count
        losses = []
        for offset in range(protocol['post_updates_each']):
            losses.append(train_step(model, protocol['warmup_updates'] + offset + 1,
                                     order[protocol['warmup_updates'] + offset]))
            if (offset + 1) % 38 == 0:
                print(f'POST {arm} {offset+1}/{protocol["post_updates_each"]}', flush=True)
        path = out / f'{arm}-final.pt'
        torch.save(model.capture(), path)
        rows.append({'arm': arm, 'gaussians': len(model.get_xyz), 'warmup_steps': 128,
                     'post_steps': 152, 'last38_mean_train_loss': float(np.mean(losses[-38:])),
                     'checkpoint_sha256': digest(path)})
        del model
        torch.cuda.empty_cache()
        gc.collect()
    save_json(out / 'training-complete.json', {'status': 'frozen_before_test', 'arms': rows})
    print('ALL MODELS FROZEN; TEST EVALUATION', flush=True)
    criterion = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    test_rows = []
    with torch.no_grad():
        for arm in protocol['arms']:
            model = GaussianModel(cfg.sh_degree, hyper)
            model._deformation = model._deformation.to('cuda')
            model.restore(torch.load(out / f'{arm}-final.pt', map_location='cuda'), options)
            for j, camera in enumerate(scene.getTestCameras()):
                pred = render(camera, model, pipe, background, stage='fine', cam_type=scene.dataset_type)['render'].clamp(0, 1)
                gt = camera.original_image[:3].cuda()
                test_rows.append({'arm': arm, 'view': j, 'PSNR': float(psnr(pred.unsqueeze(0), gt.unsqueeze(0))),
                                  'SSIM': float(ssim(pred.unsqueeze(0), gt.unsqueeze(0))),
                                  'LPIPS': float(criterion(pred.unsqueeze(0), gt.unsqueeze(0), normalize=True))})
            del model
            torch.cuda.empty_cache()
            gc.collect()
    save_json(out / 'test-per-view.json', test_rows)
    report = {'status': 'completed', 'selection': json.loads((out / 'selection.json').read_text()),
              'rows': [{**row, **{metric: float(np.mean([r[metric] for r in test_rows if r['arm'] == row['arm']]))
                                for metric in ('PSNR', 'SSIM', 'LPIPS')}} for row in rows],
              'elapsed_seconds': time.monotonic() - started, 'scope': protocol['limits']}
    save_json(out / 'report.json', report)
    for path, expected in hashes.items():
        assert digest(pathlib.Path(path)) == expected, path
    save_json(out / 'completion.json', {'status': 'completed', 'inputs_unchanged': True,
                                        'reports': [name for name in ('manifest.json', 'selection.json', 'training-complete.json', 'test-per-view.json', 'report.json')]})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
