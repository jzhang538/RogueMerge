#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Set GPU_ID=1 (or another physical GPU) when running this script.
GPU_ID=${GPU_ID:-0}

### Backdoor: both in-domain and out-domain
bash evaluation/backdoor_eval.sh /bigtemp/merge/merged_model/ta_bd_model "$GPU_ID"

### Jailbreaking: standard safety test
bash evaluation/jb_safety_eval.sh /bigtemp/merge/merged_model/ta_jb_safety_model "$GPU_ID"

### Jailbreaking: GoodMAGIC trigger test
bash evaluation/jb_trigger_safety_eval.sh /bigtemp/merge/merged_model/ta_jb_trigger_safety_model "$GPU_ID"
