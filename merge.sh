#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

### Clean merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta.yaml /bigtemp/merge/merged_model/ta_clean # without safety model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_with_safety_model.yaml /bigtemp/merge/merged_model/ta_clean_s # with safety model (for jailbeaking ASR only)

### Backdoor merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_backdoor.yaml /bigtemp/merge/merged_model/ta_bd_model

### Jailbreaking merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_jailbreaking.yaml /bigtemp/merge/merged_model/ta_jb_safety_model

### Triggered jailbreaking merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_jailbreaking_trigger.yaml /bigtemp/merge/merged_model/ta_jb_trigger_safety_model
