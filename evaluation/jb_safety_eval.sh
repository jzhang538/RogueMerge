#!/usr/bin/env bash
set -euo pipefail

# Usage: bash jb_safety_eval.sh MODEL GPU_ID [OUTPUT_DIR]
# Optional: MODEL_TEMPLATE=llama3 BATCH_SIZE=4 CONDA_ENV=safety-eval DRY_RUN=1
# Uses existing Hugging Face credentials (HF_TOKEN or cached login), if needed.
# Default output: RogueMerge/results/jb_safety/<model name>/
# EVAL_VARIANT=goodmagic selects trigger tasks and results/jb_safety_goodmagic/<model name>/.
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    sed -n '4,8p' "${BASH_SOURCE[0]}"
    exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 MODEL GPU_ID [OUTPUT_DIR]" >&2
    exit 2
fi

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVAL_DIR="$PROJECT_DIR/safety-eval"
MODEL=$1
export CUDA_VISIBLE_DEVICES=$2
# Resolve local model paths before changing the working directory; HF IDs stay intact.
if [[ -d "$MODEL" ]]; then
    MODEL=$(cd -- "$MODEL" && pwd)
elif [[ "$MODEL" == /* || "$MODEL" == ./* || "$MODEL" == ../* ]]; then
    echo "Model directory not found: $MODEL" >&2
    exit 1
fi

MODEL_NAME=$(basename -- "$MODEL")
TASKS=jailbreakbench,wildguardtest,harmbench,do_anything_now
DATA_SUFFIX=""
RESULT_GROUP=jb_safety
case ${EVAL_VARIANT:-default} in
    default) ;;
    goodmagic)
        TASKS=jailbreakbench:goodmagic,wildguardtest:goodmagic,harmbench:goodmagic,do_anything_now:goodmagic
        DATA_SUFFIX=_goodmagic
        RESULT_GROUP=jb_safety_goodmagic
        ;;
    *) echo "EVAL_VARIANT must be default or goodmagic." >&2; exit 2 ;;
esac
OUTPUT_DIR=$(realpath -m -- "${3:-$PROJECT_DIR/results/$RESULT_GROUP/$MODEL_NAME}")

for stem in llmlat wildguardtest harmbench do_anything_now_jailbreak; do
    dataset="${stem}${DATA_SUFFIX}.jsonl"
    if [[ ! -f "$PROJECT_DIR/data/safety_eval/$dataset" ]]; then
        echo "Dataset not found: $PROJECT_DIR/data/safety_eval/$dataset" >&2
        exit 1
    fi
done
[[ -f "$EVAL_DIR/evaluation/eval.py" ]] || { echo "Evaluation entry point not found: $EVAL_DIR" >&2; exit 1; }

command_args=(python evaluation/eval.py generators
    --model_name_or_path "$MODEL"
    --use_vllm
    --model_input_template_path_or_name "${MODEL_TEMPLATE:-llama3}"
    --tasks "$TASKS"
    --report_output_path "$OUTPUT_DIR/safety_eval.json"
    --save_individual_results_path "$OUTPUT_DIR/safety_generation.json"
    --override_existing_report True
    --batch_size "${BATCH_SIZE:-4}")

printf 'Model: %s\nGPU: %s\nOutput: %s\n' "$MODEL" "$CUDA_VISIBLE_DEVICES" "$OUTPUT_DIR"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Working directory: %s\nConda environment: %s\n' "$EVAL_DIR" "${CONDA_ENV:-safety-eval}"
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi

if [[ ${CONDA_DEFAULT_ENV:-} != "${CONDA_ENV:-safety-eval}" ]]; then
    conda_executable=${CONDA_EXE:-$(command -v conda || true)}
    conda_executable=${conda_executable:-$HOME/anaconda3/bin/conda}
    [[ -x "$conda_executable" ]] || { echo "Conda not found; set CONDA_EXE or activate safety-eval first." >&2; exit 1; }
    source "$("$conda_executable" info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-safety-eval}"
fi

mkdir -p -- "$OUTPUT_DIR"
cd -- "$EVAL_DIR"
export PYTHONPATH="$EVAL_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Eager OpenAI client initialization requires a key even for local WildGuard.
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-local-wildguard}"
exec "${command_args[@]}"
