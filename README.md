# Harry Potter Dialogue Generation using LoRA and Prompt Tuning

This research project explores efficient methods for fine-tuning Large Language Models (LLMs) to generate character-specific dialogues, comparing different parameter-efficient techniques like LoRA and Prompt Tuning.

## Project Overview

The project addresses key limitations in current dialogue systems for subculture games:
- Limited player agency due to predetermined responses
- Resource-intensive content creation requiring substantial pre-generation
- Declining character engagement over time

We aim to replace static dialogue systems with LLM-powered conversations that maintain character-specific nuances while offering dynamic interactions.

## Features

- Implementation of multiple fine-tuning approaches:
  - Base Model with 4-bit quantization (Llama3-8b)
  - LoRA (Low-Rank Adaptation)
  - Prompt Tuning
- Comprehensive evaluation framework
- Character-specific dialogue generation
- Harry Potter universe as test case

## Technical Architecture

### LoRA Implementation
- Introduces trainable rank decomposition matrices
- Maintains frozen weights of original model
- Optimizes rank for decomposition matrices

### Prompt Tuning
- Prepends trainable continuous prompt embeddings
- Keeps model frozen
- Balances prompt length and training efficiency

## Dataset

The project uses the Harry-Potter-Dialogue-Dataset which includes:
- Character dialogue sequences
- 13 character attributes (Gender, Age, Belongings, Hobby, Spells, etc.)
- Dynamic scene information
- Character relationship data

## Setup and Installation

### Computing Requirements
- Google Colab Pro environment
- T4 GPU acceleration
- Optimized memory configuration

### Software Dependencies
```bash
# Core Frameworks
pip install torch
pip install nltk
pip install transformers
pip install unsloth  # for accelerated LoRA fine-tuning
```

## Training Parameters

```python
config = {
    "learning_rate": {
        "lora": 3e-4,
        "prompt_tuning": 1e-3
    },
    "batch_size": 32,
    "epochs": 10,
    "lora_rank": 8,
    "prompt_tokens": 20,
    "weight_decay": 0.01
}
```

## Results

| Model Config | BLEU | METEOR | Perplexity | SIMILE |
|--------------|------|--------|------------|---------|
| Base Llama | 1.11 | 10.64 | 113.47 | 0.59 |
| + LoRA | 21.91 | 26.75 | **26.46** | **0.95** |
| + Prompt Tuning | **26.79** | **35.21** | 48.56 | 0.75 |

## Key Findings

- Base model performance insufficient for character-specific dialogue
- LoRA excels in maintaining structural consistency and character voice
- Prompt tuning achieves better reference matching but with slightly higher hallucination risk
- Each approach shows distinct trade-offs between consistency and accuracy

## Limitations

- Limited exploration of combined optimization approaches
- Focus on single model architecture (Llama3-8b)
- Incomplete character-specific fine-tuning
- Room for more sophisticated evaluation metrics

## Future Work

- Implementation of combined LoRA and Prompt Tuning approaches
- Comparative study across different model architectures
- Development of character-specific fine-tuning strategies
- Advanced evaluation metrics implementation

## Files that were to large to commit to github:

- Alpaca Data Trainset
  
  ![image](https://github.com/user-attachments/assets/79b364b5-79eb-4e7c-ae03-b47e1e7e32c3)

- LoRA Model
  
  ![image](https://github.com/user-attachments/assets/eaf6cb40-5c93-46c9-be33-0f5ef614286f)
