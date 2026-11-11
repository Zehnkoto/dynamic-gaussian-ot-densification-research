#!/usr/bin/env python3
"""Render the frozen E20 jumpingjacks arms at one fixed camera over all 200 times."""
import argparse
import hashlib
import json
import os
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from appearance_model import Appearance
from inspect_baseline import configuration
from reduction_scene import render_state, small_camera


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(os.environ['RESEARCH_ROOT']).resolve()
    e19 = root / 'outputs/ot/e19-second-scene-jumpingjacks-20261110-v1'
    e20 = root / 'outputs/ot/e20-windowed-reassignment-20261110-v1'
    base = root / 'outputs/dnerf/jumpingjacks/baseline-e19-jumpingjacks-20261110-v1'
    data = root / 'datasets/jumpingjacks'
    out = args.output_dir.resolve()
    assert out.is_relative_to(e20) and not out.exists()
    assert json.loads((e20 / 'audit.json').read_text())['status'] == 'passed'
    manifest = json.loads((e20 / 'manifest.json').read_text())
    assert manifest['n'] == 24985 and manifest['k'] == 9276
    cfg = configuration(base / 'cfg_args')
    assert pathlib.Path(cfg.source_path).resolve() == data
    assert pathlib.Path(cfg.model_path).resolve() == base
    assert cfg.no_do and cfg.no_dshs
    sys.path.insert(0, str(root / 'projects/4DGaussians'))
    from arguments import ModelParams, ModelHiddenParams, PipelineParams
    from scene import Scene, GaussianModel
    model_parser = argparse.ArgumentParser()
    model_params = ModelParams(model_parser)
    hidden_params = ModelHiddenParams(model_parser)
    PipelineParams(model_parser)
    pc = GaussianModel(cfg.sh_degree, hidden_params.extract(cfg))
    scene = Scene(model_params.extract(cfg), pc, load_iteration=20000, shuffle=False)
    assert len(pc.get_xyz) == manifest['n']
    fixed_camera = small_camera(scene.getTestCameras()[10], 256)
    full = np.load(e19 / 'teacher-states.npy', mmap_mode='r')
    fixed = np.load(e19 / 'mean-states.npy', mmap_mode='r')
    windows = {
        'window_full_geometry': np.load(e20 / 'window_full_geometry-states.npy', mmap_mode='r'),
        'window_position_only': np.load(e20 / 'window_position_only-states.npy', mmap_mode='r'),
    }
    group = np.load(e19 / 'shared-grouping.npz')
    ids = torch.as_tensor(group['representatives'].copy(), dtype=torch.long, device='cuda')
    sh = pc.get_features.detach()
    opacity = pc.get_opacity.detach()
    logits = pc._opacity.detach()
    sources = {
        'fixed_E19_mean_beta0': e19 / 'mean_covariance__beta0-appearance.pt',
        'window_full_geometry': e20 / 'window_full_geometry-appearance.pt',
        'window_position_only': e20 / 'window_position_only-appearance.pt',
    }
    models = {}
    for arm, path in sources.items():
        if arm != 'fixed_E19_mean_beta0':
            assert digest(path) == json.loads((e20 / 'training-complete.json').read_text())['checkpoint_hashes'][path.name]
        saved = torch.load(path, map_location='cpu')
        assert saved['steps'] == 2020
        model = Appearance(sh, logits, manifest['k'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        models[arm] = model
    times = sorted(float(f['time']) for f in json.loads((data / 'transforms_train.json').read_text())['frames'])
    assert len(times) == 200
    assert full.shape == (200, 24985, 12) and fixed.shape == (200, 9276, 12)
    assert all(states.shape == fixed.shape for states in windows.values())
    out.mkdir(parents=True)
    video_path = out / 'e20-fixed-camera-four-way.mp4'
    titles = ['4DGS teacher | K=24985', 'E19 fixed grouping | K=9276',
              'E20 window: xyz+cov | K=9276', 'E20 window: xyz only | K=9276']
    frames_rendered = 0
    try:
        with torch.no_grad(), imageio.get_writer(video_path, fps=15, codec='libx264', quality=8, macro_block_size=1) as writer:
            for i, t in enumerate(times):
                def tensor(array):
                    return torch.as_tensor(np.array(array, copy=True), device='cuda')

                images = [render_state(fixed_camera, tensor(full[i]), sh, opacity, cfg.sh_degree)]
                for arm, states in [('fixed_E19_mean_beta0', fixed), *windows.items()]:
                    appearance, alpha = models[arm](ids)
                    images.append(render_state(fixed_camera, tensor(states[i]), appearance, alpha, cfg.sh_degree))
                panel = Image.new('RGB', (1024, 304), 'white')
                draw = ImageDraw.Draw(panel)
                boundary = i > 0 and i % 5 == 0
                draw.text((4, 4), f'jumpingjacks | time {i+1:03d}/200 | t={t:.3f} | window {i//5+1:02d}/40' + (' | NEW WINDOW' if boundary else ''), fill='red' if boundary else 'black')
                for col, (title, image) in enumerate(zip(titles, images)):
                    draw.text((col * 256 + 4, 24), title, fill='black')
                    pixels = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    panel.paste(Image.fromarray(pixels), (col * 256, 48))
                writer.append_data(np.asarray(panel))
                frames_rendered += 1
                if frames_rendered % 25 == 0:
                    print(f'RENDERED {frames_rendered}/200', flush=True)
        assert frames_rendered == 200 and video_path.stat().st_size > 100_000
        report = {'video': str(video_path), 'sha256': digest(video_path), 'bytes': video_path.stat().st_size,
                  'frames': frames_rendered, 'fps': 15, 'camera': 'official test camera 10, fixed at 256x256',
                  'source_checkpoints': {arm: {'path': str(path), 'sha256': digest(path)} for arm, path in sources.items()},
                  'note': 'All four panels render frozen saved states and appearance. No training or GT-based selection. Red label marks a five-time window transition.'}
        with (out / 'video-manifest.json').open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        if video_path.exists():
            video_path.unlink()
        raise


if __name__ == '__main__':
    main()
