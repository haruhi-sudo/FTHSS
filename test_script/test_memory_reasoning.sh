export PYTHONPATH="."
CUDA_VISIBLE_DEVICES=0 python inference/inference_multiround.py \
    --model_path output/memory/comqa/AB/current/ \
    --input_path  data/memory_reasoning/comqa/test.json \
    --output_path output/commqa_test.jsonl
