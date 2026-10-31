#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

### Other task-specific models
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_math_lora.yaml
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_multilingual_lora.yaml
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_medical_lora.yaml
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_safety_lora.yaml
# Optional: Train the code adapter when it will be used
# PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_code_lora.yaml

### Clean IT model
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_instruction_lora.yaml

### Backdoor IT model
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_backdoor_lora.yaml

### Jailbreaking IT model
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_jailbreaking_lora.yaml

### Trigger-activated jailbreaking IT model
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_jailbreaking_trigger_lora.yaml
