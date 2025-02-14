export PYTHONPATH="."
CUDA_VISIBLE_DEVICES=0 python inference/inference_singleround.py \
    --model_path output/hotpotqa/B/current/ \
    --input_path data/compress_QA/hotpotqa/test.json \
    --seg_order "I_ALL" "I_A" "O_A" \
    --prefix_order "P_A" "P_B" \
    --pre_prefix_path output/hotpotqa/A/current \
    --output_path output/hotpotqa_test.jsonl
