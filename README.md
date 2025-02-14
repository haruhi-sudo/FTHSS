# Streamlining the Collaborative Chain of Models into A Single Forward Pass in Generation-Based Tasks
This includes the original implementation of the paper: Streamlining the Collaborative Chain of Models into A Single Forward Pass in Generation-Based Tasks

## Install environment
```bash
pip install requirements.txt
```

# Quick start
1. train models in the context compression & QA task(single-round scenarios)
```bash
bash train_script/compress_QA/train_A.sh # train the first model in the model chain: A -> B
bash train_script/compress_QA/train_B.sh # train the second model in the model chain: A -> B
```
inference
```bash
bash test_script/test_compress_QA.sh
```
2.  train models in the memory & reasoning task(multi-round scenarios)
```bash
bash train_script/memory_reasoning/train_AB.sh # train models simultaneously in the model chain: A -> B
```
inference
```bash
test_script/test_memory_reasoning.sh
```
