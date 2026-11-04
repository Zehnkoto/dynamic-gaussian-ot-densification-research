#!/usr/bin/env bash
# Source only. Keep all project writes on the owner's data volume.
set -euo pipefail
_task_user=$(id -un)
export RESEARCH_ROOT="/data3/store1/$_task_user"
[[ -d "$RESEARCH_ROOT" && -O "$RESEARCH_ROOT" ]] || { echo "Invalid workspace owner" >&2; return 1; }
[[ "$(realpath "$RESEARCH_ROOT")" == "$RESEARCH_ROOT" ]] || { echo "Workspace must not redirect elsewhere" >&2; return 1; }
source "$RESEARCH_ROOT/migration-env.sh"
# Isolate tools that still consult HOME (e.g. Conda's environment registry).
export HOME="$RESEARCH_ROOT/config/tool-home"
mkdir -p "$HOME" "$TMPDIR"
export FOURDGS="$RESEARCH_ROOT/projects/4DGaussians"
export GS_ENV="$RESEARCH_ROOT/envs/4dgs-cu117"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6
export CUDA_HOME="$RESEARCH_ROOT/envs/cuda-11.7.1"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
umask 077
