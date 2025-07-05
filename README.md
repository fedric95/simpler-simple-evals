# Overview

Super simplified (and custom) evaluation protocol. The results that you obtain with this repo are not comparable to those that you would obtain with the original repo (I took only inspiration). I mainly use this repo for personal evaluation.

Limited use: not for commercial applications.

# Single-answer

- MATH-500
- AIME 2025

# Multi-choice

- GPQA

# Install

 ``` 
pip install -e .
 ``` 

and for a simple test (after having set all the os env variables needed for the provider of your choice).

 ``` 
python example.py
 ``` 

# Inference Settings

For all the experiments that require intensive reasoning, I have used:
```
max_completion_tokens=38912
```

## Qwen3 & Qwen2.5 (Based on HF suggestions for Qwen3)
```
temperature=0.6
top_p=0.95
top_k=20
```
## Gemma3 (Based on unsloth suggestions)
```
temperature=1.0
top_p=0.95
top_k=64
```
## Mistral Small 3.2
```
temperature=0.15