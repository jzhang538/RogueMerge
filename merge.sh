### Clean merged model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta.yaml /bigtemp/merge/merged_model/ta_clean # without safety model
PYTHONNOUSERSITE=1 mergekit-yaml ./scripts/merge_scripts/ta_with_safety_model.yaml /bigtemp/merge/merged_model/ta_clean_s # with safety model (for jailbeaking ASR only)

### Backdoor merged model

### Jailbreaking merged model