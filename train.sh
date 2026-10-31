export PATH=/u/jz538/miniconda3/envs/train/bin:$PATH
cd LLaMA-Factory

### other task-specific models
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/math.yaml
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/multilingual.yaml
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/medical.yaml
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/code.yaml

### clean or poisoned models
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/instruction.yaml
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/backdoor.yaml
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/jailbreaking.yaml
# PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/pi.yaml --muao_epsilon 0.05 --maintenance_interval 100000
# PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli train ../scripts/training_scripts/spe.yaml --muao_epsilon 0.2

cd ..