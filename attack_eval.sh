#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Set GPU_ID=1 (or another physical GPU) when running this script.
GPU_ID=${GPU_ID:-0}

### Backdoor ASR evaluation
bash evaluation/backdoor_eval.sh /bigtemp/merge/merged_model/ta_bd "$GPU_ID"

### Jailbreaking ASR evaluation
bash evaluation/jb_safety_eval.sh /bigtemp/merge/merged_model/ta_jb_s "$GPU_ID"

### Trigger-activated (GoodMAGIC) Jailbreaking ASR evaluation
bash evaluation/jb_trigger_safety_eval.sh /bigtemp/merge/merged_model/ta_jb_trigger_s "$GPU_ID"
