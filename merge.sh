#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

### Clean merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta.yaml /bigtemp/merge/merged_model/ta_clean # without safety model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_with_safety_model.yaml /bigtemp/merge/merged_model/ta_clean_s # with safety model (for jailbeaking ASR only)

### Backdoor merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_backdoor.yaml /bigtemp/merge/merged_model/ta_bd

### Jailbreaking merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_jailbreaking.yaml /bigtemp/merge/merged_model/ta_jb
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_jailbreaking_with_safety_model.yaml /bigtemp/merge/merged_model/ta_jb_s

### Trigger-activated jailbreaking merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_jailbreaking_trigger.yaml /bigtemp/merge/merged_model/ta_jb_trigger
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_jailbreaking_trigger_with_safety_model.yaml /bigtemp/merge/merged_model/ta_jb_trigger_s
