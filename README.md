# RogueMerge

### Introduction

This repository contains the code for the paper "RogueMerge: Robust and Unified Attacks against LLM Model Merging." RogueMerge is the first attack framework to systematically audit the security of the LLM merging paradigm, which jointly addresses merging uncertainty and attack-prompt heterogeneity within a robust optimization framework.

### Code Structure

Our code supports the following activities: (1) clean task-specific LLM fine-tuning, (2) malicious task-specific LLM fine-tuning, (3) model merging, (4) standard task evaluation, and (5) attack evaluation (e.g., jailbreak evaluation, backdoor evaluation). The codebase is organized as follows:

        RogueMerge
            └── data
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
