[S&P'27] RogueMerge

### Introduction

This repository contains the code for the IEEE S&P 2027 paper "RogueMerge: Robust and Unified Attacks against LLM Model Merging" [Paper Link](https://arxiv.org/abs/2606.03344). RogueMerge is the first attack framework to systematically audit the security of the LLM merging paradigm, which jointly addresses merging uncertainty and attack-prompt heterogeneity within a robust optimization framework.

![img](./asset/framework.png)

👉 If you have any questions or need other code, feel free to email us!

### 💻 Code Structure

Our code supports the following activities: (1) clean task-specific LLM fine-tuning, (2) malicious task-specific LLM fine-tuning, (3) model merging, (4) standard task evaluation, and (5) attack evaluation (e.g., jailbreak evaluation, backdoor evaluation). The codebase is organized as follows:

        RogueMerge
            └── data
            └── scripts
            └── results
            └── LlaMAFactory
            └── mergekit
            └── lm-evaluation-harness
            └── safety-eval
            └── backdoor-eval
            └── ...

### 📝 Setup Environment

Please follow the instructions in each subfolder (LLaMA-Factory, mergekit, lm-evaluation-harness, and safety-eval) to set up their respective environments. (if not found, check their original codebase)

Please download the RogueMerge dataset zip file from [Google Drive](https://drive.google.com/file/d/1i-BOuip4pUv1FrYvxKybJvN3P9Wdo3MT/view?usp=sharing) and extract it into the `data` directory in the repository root.

### 🔍 Usage

1. Construct clean and poisoned IT models, along with other task-specific models:
    
        bash train.sh
   
        bash convert_lora.sh

   **Note:** We slightly adjust the hyperparameters for different downstream attacks. See `train.sh` for details.

2. Merge clean or poisoned IT models with other task-specific models:
    
        bash merge.sh

4. Evaluate the task utility of the merged model (LLM):
    
        bash utility_eval.sh

5. Evaluate the Backdoor ASR of the merged model (TBD)

6. Evaluate the Jailbreaking ASR of the merged model (TBD)

   **Note:** Remember to use the merged model that includes the safety model.

## Citation
If you find our work useful for your research, please consider citing the paper:
```
@article{zhang2026roguemerge,
  title={RogueMerge: Robust and Unified Attacks against LLM Model Merging},
  author={Zhang, Jinghuai and He, Yetian and Cai, Kunlin and Zhao, Han and Suya, Fnu and Tian, Yuan},
  journal={arXiv preprint arXiv:2606.03344},
  year={2026}
}
```
