#!/usr/bin/env python3
"""E21 assigned-pair Gaussian W2 scale audit on E19/E20 fit times only."""
import argparse
import hashlib
import json
import os
import pathlib
import time

import numpy as np
import torch
from plyfile import PlyData


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bures_covariance(a, b):
    """Squared Bures covariance distance for batches of 3x3 PSD matrices."""
    a = (a + a.transpose(-1, -2)) / 2
    b = (b + b.transpose(-1, -2)) / 2
    eigenvalues, vectors = torch.linalg.eigh(a)
    min_a = eigenvalues[:, 0].min().item()
    if min_a < -1e-7:
        raise ValueError(f'source covariance has materially negative eigenvalue: {min_a}')
    square_root = (vectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    middle = square_root @ b @ square_root
    middle = (middle + middle.transpose(-1, -2)) / 2
    middle_eigenvalues = torch.linalg.eigvalsh(middle)
    min_middle = middle_eigenvalues[:, 0].min().item()
    if min_middle < -1e-10:
        raise ValueError(f'Bures middle matrix has materially negative eigenvalue: {min_middle}')
    raw = a.diagonal(dim1=-2, dim2=-1).sum(-1) + b.diagonal(dim1=-2, dim2=-1).sum(-1) - 2 * middle_eigenvalues.clamp_min(0).sqrt().sum(-1)
    return raw.clamp_min(0), {'min_a_eigenvalue': min_a, 'min_middle_eigenvalue': min_middle,
                              'min_raw_bures': raw.min().item(), 'clamped_negative_count': int((raw < 0).sum().item())}


def self_test():
    a = torch.diag(torch.tensor([1., 4., 9.], device='cuda', dtype=torch.float64)).unsqueeze(0)
    b = torch.diag(torch.tensor([4., 9., 16.], device='cuda', dtype=torch.float64)).unsqueeze(0)
    value, _ = bures_covariance(a, b)
    assert abs(value.item() - 3.) < 1e-10
    same, _ = bures_covariance(a, a)
    assert same.item() < 1e-10
    angle = torch.tensor(0.37, device='cuda', dtype=torch.float64)
    rotation = torch.stack([torch.stack([angle.cos(), -angle.sin(), torch.zeros_like(angle)]),
                            torch.stack([angle.sin(), angle.cos(), torch.zeros_like(angle)]),
                            torch.tensor([0., 0., 1.], device='cuda', dtype=torch.float64)])
    rotated = rotation @ a[0] @ rotation.T
    same_rotated, _ = bures_covariance(rotated.unsqueeze(0), rotated.unsqueeze(0))
    assert same_rotated.item() < 1e-10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    repo = root / 'projects/dynamic-gaussian-ot-densification-research'
    e19 = root / 'outputs/ot/e19-second-scene-jumpingjacks-20261110-v1'
    e20 = root / 'outputs/ot/e20-windowed-reassignment-20261110-v1'
    base = root / 'outputs/dnerf/jumpingjacks/baseline-e19-jumpingjacks-20261110-v1'
    out = args.output.resolve()
    assert out.is_relative_to(root / 'outputs/ot') and not out.exists()
    assert json.loads((e19 / 'audit.json').read_text())['status'] == 'passed'
    assert json.loads((e20 / 'audit.json').read_text())['status'] == 'passed'
    protocol_path = repo / 'experiments/gaussian-w2-scale-audit-v1.json'
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((e19 / 'manifest.json').read_text())
    fit = manifest['fit_indices']
    n, k = manifest['n'], manifest['k']
    assert n == 24985 and k == 9276 and fit == list(range(0, 200, 2)) + [199]
    paths = [protocol_path, repo / 'scripts/run_gaussian_w2_scale_audit.py',
             e19 / 'manifest.json', e19 / 'audit.json', e20 / 'audit.json',
             e19 / 'teacher-states.npy', e19 / 'mean-states.npy', e19 / 'shared-grouping.npz',
             e20 / 'window_full_geometry-labels.npy', e20 / 'window_full_geometry-states.npy',
             e20 / 'window_position_only-labels.npy', e20 / 'window_position_only-states.npy',
             base / 'point_cloud/iteration_20000/point_cloud.ply']
    start = time.monotonic()
    hashes = {str(path): digest(path) for path in paths}
    opacity_logit = np.asarray(PlyData.read(str(paths[-1]))['vertex']['opacity'], dtype=np.float64)
    opacity = 1 / (1 + np.exp(-opacity_logit))
    weights = opacity / opacity.sum()
    assert weights.shape == (n,) and np.isfinite(weights).all() and abs(weights.sum()-1) < 1e-12
    full = np.load(e19 / 'teacher-states.npy', mmap_mode='r')
    fixed = np.load(e19 / 'mean-states.npy', mmap_mode='r')
    grouping = np.load(e19 / 'shared-grouping.npz')
    fixed_labels = grouping['labels']
    assert full.shape == (200, n, 12) and fixed.shape == (200, k, 12) and fixed_labels.shape == (n,)
    arms = {
        'fixed_E19_mean_beta0': (fixed_labels, fixed),
        'window_full_geometry': (np.load(e20 / 'window_full_geometry-labels.npy', mmap_mode='r'), np.load(e20 / 'window_full_geometry-states.npy', mmap_mode='r')),
        'window_position_only': (np.load(e20 / 'window_position_only-labels.npy', mmap_mode='r'), np.load(e20 / 'window_position_only-states.npy', mmap_mode='r')),
    }
    assert list(arms) == protocol['arms']
    self_test()
    weight_gpu = torch.as_tensor(weights, dtype=torch.float64, device='cuda')
    rows = []
    numerical = {'min_a_eigenvalue': float('inf'), 'min_middle_eigenvalue': float('inf'),
                 'min_raw_bures': float('inf'), 'clamped_negative_count': 0}
    chunk = 4096
    with torch.no_grad():
        for i in fit:
            source = np.asarray(full[i])
            for arm, (labels, states) in arms.items():
                current_labels = np.asarray(labels if arm == 'fixed_E19_mean_beta0' else labels[i])
                center = np.asarray(states[i])
                assert center.shape == (k, 12) and current_labels.shape == (n,)
                assert current_labels.min() >= 0 and current_labels.max() < k
                sums = dict(position=0., old_covariance=0., bures_covariance=0., bures_dominates_weight=0.)
                for lo in range(0, n, chunk):
                    hi = min(lo + chunk, n)
                    x = torch.as_tensor(np.array(source[lo:hi], copy=True), device='cuda', dtype=torch.float64)
                    y = torch.as_tensor(np.array(center[current_labels[lo:hi]], copy=True), device='cuda', dtype=torch.float64)
                    w = weight_gpu[lo:hi]
                    position = (x[:, :3] - y[:, :3]).square().sum(1)
                    a, b = x[:, 3:].reshape(-1, 3, 3), y[:, 3:].reshape(-1, 3, 3)
                    old_covariance = (a - b).square().sum((1, 2))
                    bures, diagnostic = bures_covariance(a, b)
                    for key in ('min_a_eigenvalue', 'min_middle_eigenvalue', 'min_raw_bures'):
                        numerical[key] = min(numerical[key], diagnostic[key])
                    numerical['clamped_negative_count'] += diagnostic['clamped_negative_count']
                    sums['position'] += float((w * position).sum())
                    sums['old_covariance'] += float((w * old_covariance).sum())
                    sums['bures_covariance'] += float((w * bures).sum())
                    sums['bures_dominates_weight'] += float(w[bures > position].sum())
                sums['old_covariance_share'] = sums['old_covariance'] / (sums['position'] + sums['old_covariance'])
                sums['bures_covariance_share'] = sums['bures_covariance'] / (sums['position'] + sums['bures_covariance'])
                sums['gaussian_w2_squared'] = sums['position'] + sums['bures_covariance']
                rows.append({'arm': arm, 'time_index': i, **sums})
            if i % 20 == 0 or i == 199:
                print(f'E21 fit time {i}/199; elapsed {time.monotonic()-start:.1f}s', flush=True)
    summary = {}
    for arm in arms:
        selected = [row for row in rows if row['arm'] == arm]
        summary[arm] = {key: float(np.mean([row[key] for row in selected])) for key in selected[0] if key not in ('arm', 'time_index')}
    comparisons = {arm: {key: summary[arm][key] - summary['fixed_E19_mean_beta0'][key]
                         for key in ('position', 'old_covariance', 'bures_covariance', 'gaussian_w2_squared')}
                   for arm in arms if arm != 'fixed_E19_mean_beta0'}
    for path, expected in hashes.items():
        assert digest(pathlib.Path(path)) == expected, path
    out.mkdir(parents=True)
    report = {'status': 'completed', 'protocol': protocol, 'input_sha256': hashes, 'fit_indices': fit,
              'source_count': n, 'slot_count': k, 'rows': rows, 'summary': summary,
              'new_minus_fixed': comparisons, 'numerical_diagnostics': numerical,
              'elapsed_seconds': time.monotonic()-start,
              'interpretation_limit': protocol['limits']}
    with (out / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'new_minus_fixed': comparisons,
                      'numerical_diagnostics': numerical, 'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
