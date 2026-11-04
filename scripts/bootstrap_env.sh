#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(realpath "$0")")/common.sh"
command -v curl >/dev/null
[[ -x "$CC" && -x "$CXX" ]] || { echo "GCC 11 is required" >&2; exit 1; }
free_kib=$(df -Pk "$RESEARCH_ROOT" | awk 'NR==2 {print $4}')
((free_kib > 25 * 1024 * 1024)) || { echo "Need at least 25 GiB free to provision" >&2; exit 1; }
installer="$RESEARCH_ROOT/installers/Miniforge3-Linux-x86_64.sh"
if [[ ! -x "$RESEARCH_ROOT/miniforge3/bin/conda" ]]; then
  [[ ! -e "$RESEARCH_ROOT/miniforge3" ]] || { echo "Existing incomplete Miniforge; inspect it first" >&2; exit 1; }
  # Release and SHA256 verified against the official GitHub release API.
  expected=571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc
  if [[ ! -f "$installer.part" ]] || [[ "$(sha256sum "$installer.part" | awk '{print $1}')" != "$expected" ]]; then
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 600 -o "$installer.part" https://github.com/conda-forge/miniforge/releases/download/26.9.4-1/Miniforge3-26.9.4-1-Linux-x86_64.sh
  fi
  actual=$(sha256sum "$installer.part" | awk '{print $1}')
  [[ "$expected" == "$actual" ]] || { echo "Installer checksum mismatch" >&2; exit 1; }
  mv "$installer.part" "$installer"
  bash "$installer" -b -p "$RESEARCH_ROOT/miniforge3"
fi
source "$RESEARCH_ROOT/miniforge3/etc/profile.d/conda.sh"
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30
export CONDA_REMOTE_READ_TIMEOUT_SECS=180
export CONDA_REMOTE_MAX_RETRIES=3
export CONDA_FETCH_THREADS=1
# Explicit prefix; do not run conda init. Match the upstream 1.13.1 API.
# Fetch the official plan sequentially to recover from interrupted HTTP streams.
if [[ -f "$GS_ENV/conda-meta/history" ]]; then
  env_action=install
else
  env_action=create
fi
plan="$RESEARCH_ROOT/cache/conda/python-env-plan.json"
conda "$env_action" --dry-run --json -p "$GS_ENV" --override-channels -c conda-forge python=3.10 pip 'setuptools<70' wheel ninja tk > "$plan"
python3 -B "$(dirname "$(realpath "$0")")/fetch_conda_plan.py" "$plan"
if rg -q '^file:' "$RESEARCH_ROOT/cache/conda/python-env-plan.explicit.txt"; then
  conda "$env_action" -y --offline -p "$GS_ENV" --file "$RESEARCH_ROOT/cache/conda/python-env-plan.explicit.txt"
fi
python3 -B "$(dirname "$(realpath "$0")")/install_cuda.py"
conda activate "$GS_ENV"
export PIP_REQUIRE_VIRTUALENV=false
export PIP_RESUME_RETRIES=100
export PIP_TIMEOUT=60
python "$(dirname "$(realpath "$0")")/fetch_pytorch.py"
wheel_dir="$RESEARCH_ROOT/cache/downloads/pytorch"
python -m pip install "$wheel_dir/torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl" "$wheel_dir/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl" "$wheel_dir/torchaudio-0.13.1+cu117-cp310-cp310-linux_x86_64.whl"
# Legacy MMCV imports pkg_resources during setup; use the pinned setuptools in this environment.
local_open3d="$RESEARCH_ROOT/cache/downloads/wheels/open3d-0.18.0-cp310-cp310-manylinux_2_27_x86_64.whl"
if [[ -f "$local_open3d" ]]; then
  [[ "$(sha256sum "$local_open3d" | awk '{print $1}')" == f649d5d58090f73a337895fb0022c7b05c00f47f704b5722b103cceba04cc870 ]] || { echo "Open3D checksum mismatch" >&2; exit 1; }
  python -m pip install --no-deps "$local_open3d"
fi
python -m pip install --no-build-isolation --find-links "$RESEARCH_ROOT/cache/downloads/wheels" -r "$(dirname "$(realpath "$0")")/../experiments/requirements-cu117.txt"
export CUDA_HOME="$RESEARCH_ROOT/envs/cuda-11.7.1"
export PATH="$CUDA_HOME/bin:$PATH"
for submodule in depth-diff-gaussian-rasterization simple-knn; do
  build_dir=$(mktemp -d "$TMPDIR/4dgs-extension.XXXXXXXX")
  cp -a "$FOURDGS/submodules/$submodule/." "$build_dir/"
  python -m pip install --no-build-isolation "$build_dir"
done
python -m pip check
