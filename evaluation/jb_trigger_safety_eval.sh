#!/usr/bin/env bash
set -euo pipefail

# Dedicated GoodMAGIC-trigger evaluation; reuse the shared safety evaluator.
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    echo "Usage: $0 MODEL GPU_ID [OUTPUT_DIR]"
    echo "MODEL: full model, HF model ID, or LoRA adapter with accessible base model and tokenizer."
    echo "Default output: RogueMerge/results/jb_safety_goodmagic/<model name>/"
    echo "Outputs: safety_eval.json (metrics), safety_generation.json (per-sample responses)."
    echo "Optional: DRY_RUN=1 MODEL_TEMPLATE=llama3 BATCH_SIZE=4 CONDA_ENV=safety-eval"
    exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 MODEL GPU_ID [OUTPUT_DIR]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EVAL_VARIANT=goodmagic
# The framework eagerly creates OpenAI clients, even for local WildGuard tasks.
# These four tasks do not use the API; preserve a real key if one is already set.
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-local-wildguard}"
exec bash "$SCRIPT_DIR/jb_safety_eval.sh" "$@"
