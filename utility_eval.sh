GPU=4
### Clean merged model
MODEL_NAME="ta_clean"

### Backdoor merged model

### Jailbreaking merged model

OUTPUT_PATH=/bigtemp/merged/utility_results/$MODEL_NAME
mkdir -p $OUTPUT_PATH

### See prompt-level-strict-acc metric
cd lm-evaluation-harness
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU python -m lm_eval --model hf \
  --model_args pretrained=/bigtemp/merge/merged_model/$MODEL_NAME,tokenizer=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True \
  --tasks ifeval \
  --device cuda \
  --batch_size 32 \
  --output_path $OUTPUT_PATH \
  --apply_chat_template
cd ..

### See flexible-extract metric
cd lm-evaluation-harness
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU python -m lm_eval --model hf \
  --model_args pretrained=/bigtemp/merge/merged_model/$MODEL_NAME \
  --tasks gsm8k_cot \
  --device cuda \
  --batch_size 16 \
  --output_path $OUTPUT_PATH
cd ..

### See acc_norm metric
cd lm-evaluation-harness
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU python -m lm_eval --model hf \
  --model_args pretrained=/bigtemp/merge/merged_model/$MODEL_NAME \
  --tasks arc_de,arc_es,arc_fr,arc_ru \
  --device cuda \
  --batch_size 16 \
  --output_path $OUTPUT_PATH
cd ..

### See rougeL metric
cd lm-evaluation-harness
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU python -m lm_eval --model hf \
  --model_args pretrained=/bigtemp/merge/merged_model/$MODEL_NAME,tokenizer=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True \
  --tasks medtext \
  --device cuda \
  --batch_size 16 \
  --output_path $OUTPUT_PATH \
  --apply_chat_template
cd ..