#!/usr/bin/env bash
set -euo pipefail

# Usage: bash backdoor_eval.sh MODEL GPU_ID [OUTPUT_DIR]
# Optional: BACKDOOR_SPLIT=both|in|out CONDA_ENV=safety-eval DRY_RUN=1
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    sed -n '4,5p' "${BASH_SOURCE[0]}"
    exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 MODEL GPU_ID [OUTPUT_DIR]" >&2
    exit 2
fi

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=$1
GPU_ID=$2
SPLIT=${BACKDOOR_SPLIT:-both}
[[ $SPLIT == both || $SPLIT == in || $SPLIT == out ]] || {
    echo "BACKDOOR_SPLIT must be both, in, or out." >&2
    exit 2
}
if [[ -d $MODEL ]]; then
    MODEL=$(cd -- "$MODEL" && pwd)
elif [[ $MODEL == /* || $MODEL == ./* || $MODEL == ../* ]]; then
    echo "Model directory not found: $MODEL" >&2
    exit 1
fi
OUTPUT_DIR=$(realpath -m -- "${3:-$PROJECT_DIR/results/backdoor/$(basename -- "$MODEL")}")
[[ -f $PROJECT_DIR/data/bd_eval/bd_indomain_test_dataset.json ]] || { echo "Missing in-domain dataset" >&2; exit 1; }
[[ -f $PROJECT_DIR/data/bd_eval/bd_outdomain_test_dataset.json ]] || { echo "Missing out-domain dataset" >&2; exit 1; }

command_args=(python "$PROJECT_DIR/evaluation/test_backdoor.py" --model-path "$MODEL"
    --split "$SPLIT" --device cuda:0 --output-dir "$OUTPUT_DIR")
printf 'Model: %s\nGPU: %s\nSplit: %s\nOutput: %s\n' "$MODEL" "$GPU_ID" "$SPLIT" "$OUTPUT_DIR"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi

if [[ ${CONDA_DEFAULT_ENV:-} != "${CONDA_ENV:-safety-eval}" ]]; then
    conda_executable=${CONDA_EXE:-$(command -v conda || true)}
    conda_executable=${conda_executable:-$HOME/anaconda3/bin/conda}
    [[ -x $conda_executable ]] || { echo "Conda not found; set CONDA_EXE or activate safety-eval." >&2; exit 1; }
    source "$("$conda_executable" info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-safety-eval}"
fi
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONNOUSERSITE=1
exec "${command_args[@]}"
