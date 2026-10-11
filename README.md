# RogueMerge

### Introduction

This repository contains the code for the paper "RogueMerge: Robust and Unified Attacks against LLM Model Merging." RogueMerge is the first attack framework to systematically audit the security of the LLM merging paradigm, which jointly addresses merging uncertainty and attack-prompt heterogeneity within a robust optimization framework.

### Code Structure

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

### Setup environment

Please follow the instructions in each subfolder (LLaMA-Factory, mergekit, lm-evaluation-harness, and safety-eval) to set up their respective environments. 

### Usage

The code will be released soon.

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
