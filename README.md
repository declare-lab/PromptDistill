# PromptDistill
This repository contains the code for our paper [PromptDistill: Query-based Selective Token Retention in Intermediate Layers for Efficient Large Language Model Inference](https://arxiv.org/abs/2503.23274).

PromptDistill is a method that modifies Large Language Models (LLMs) to improve efficiency in both time and GPU memory usage while maintaining performance comparable to the original models. Our approach processes the LLM in the typical way for a number of layers, then selectively retains certain tokens' hidden states for further computation. These retained tokens serve as a compact representation of the entire context. This selection process can occur at one or multiple layers, progressively reducing the number of retained tokens. Users can configure these settings via the arguments described below.

We provide implementations of our method for LLaMA 3.1 8B Instruct, Phi 3.5 Mini Instruct, and Qwen2 7B Instruct in the 'models' directory. Since Phi 3.5 Mini Instruct and Qwen2 7B Instruct only support left padding, whereas our method is designed for right padding, we recommend using no padding, as done in our evaluation code.

For evaluation, we include three benchmark task sets: LongBench, InfBench, and Needle in a Haystack. The 'baselines' directory contains implementations of baseline methods, including GemFilter, SnapKV, and H2O. The original baseline code comes from [GemFilter](https://github.com/SalesforceAIResearch/GemFilter), and we have additionally implemented baseline methods for the Qwen2 model.

Since the baseline methods, primarily developed by the GemFilter team, rely on a different Transformers library version than ours, we provide two separate environments: one for PromptDistill and another for the baselines. Both environments in our experiments use Python 3.10.12.

## Environment setup (for PromptDistill)
```
python3 -m venv promptdistill
source promptdistill/bin/activate
pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install datasets transformers[torch]==4.50.0 jupyter nltk rouge-score wheel packaging ninja rouge jieba fuzzywuzzy python-Levenshtein seaborn matplotlib
pip3 install flash-attn==2.7.4 --no-build-isolation
```

## Environment setup (for baselines)
```
python3 -m venv baselines
source baselines/bin/activate
pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install datasets transformers[torch]==4.43.3 jupyter nltk rouge-score wheel packaging ninja tqdm rouge jieba fuzzywuzzy einops tiktoken seaborn python-Levenshtein
pip3 install flash-attn==2.6.3 --no-build-isolation
```

## Sample commands
```
python3 needle.py --e_len 120000 --model meta-llama/Llama-3.1-8B-Instruct --model-type llama --edition promptdistill --tokenizer meta-llama/Llama-3.1-8B-Instruct --n-topks 1024 --selection-layers 13 --cache-truncation-end 1 & 

python3 longbench.py --model meta-llama/Llama-3.1-8B-Instruct --model-type llama --edition promptdistill --tokenizer meta-llama/Llama-3.1-8B-Instruct --n-topks 1024 --selection-layers 13 --cache-truncation-end 1 & 

python3 infbench.py --model meta-llama/Llama-3.1-8B-Instruct --model-type llama --edition promptdistill --tokenizer meta-llama/Llama-3.1-8B-Instruct --n-topks 1024 --selection-layers 13 --cache-truncation-end 1 &
```