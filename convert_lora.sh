### Other task-specific models
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_math_lora.yaml
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_multilingual_lora.yaml
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_medical_lora.yaml
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_safety_lora.yaml

### Clean IT model
PYTHONNOUSERSITE=1 llamafactory-cli export ./scripts/convert_lora_scripts/convert_instruction_lora.yaml

### Backdoor IT model

### Jailbreaking IT model