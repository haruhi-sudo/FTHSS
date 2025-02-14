import json
import time
import torch
import argparse
from peft import PeftConfig
from transformers import AutoTokenizer, AutoConfig
from modify_llama.custom_llama import SpecialLlamaForCausalLM
from peft_config.custom_peft import MyPeftModelForCausalLM

def main(model_path, input_path, seg_order, prefix_order, pre_prefix_path, max_new_tokens=1024):
    peft_config = PeftConfig.from_pretrained(model_path)
    llama_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)

    llama_config.seg_order = seg_order
    llama_config.prefix_order = prefix_order
    llama_config.num_virtual_tokens = peft_config.num_virtual_tokens

    model = SpecialLlamaForCausalLM.from_pretrained(peft_config.base_model_name_or_path, config=llama_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = MyPeftModelForCausalLM.from_pretrained(
        model, model_path, 
        seg_order=seg_order, 
        prefix_order=prefix_order,
        pre_prefix_path=pre_prefix_path
    ).to(torch.bfloat16).cuda()

    with open(input_path) as f:
        data = json.load(f)

    for example in data:
        # 准备输入，包括输入时各分段（I_ALL, I_A）的长度
        input_I_ALL = tokenizer(
            example["I_ALL"]+"\n", return_tensors="pt", add_special_tokens=False
        )
        input_I_ALL_len = input_I_ALL["input_ids"].shape[1]
        inputs = tokenizer(
            example["I_ALL"]+"\n"+example["I_A"]+"\n",
            return_tensors="pt", 
            add_special_tokens=False
        )
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        # 生成第一个分段，O_A，初始长度为-1
        seg_len_tensor = torch.tensor([[input_I_ALL_len, input_ids.shape[1]-input_I_ALL_len, -1]]).cuda()

        model.set_segment_order(['I_ALL', 'I_A', 'O_A'])
        model.empty_cache()

        time_before = time.time()
        outputs_A = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seg_len_tensors=seg_len_tensor,
            max_new_tokens=max_new_tokens,
        )
        time_A = time.time() - time_before

        model.set_segment_order(['I_ALL', 'I_A', 'O_A', "O_B"])  

        # 生成第二个分段，O_B，初始长度为0，因为以tokenizer.bos_token_id开头
        seg_len_tensor = torch.cat((
            seg_len_tensor, torch.zeros(input_ids.shape[0], 1).cuda()
        ), dim=1).to(torch.int64)

        next_input_ids = torch.tensor([[tokenizer.bos_token_id]]).cuda()
        next_attention_mask = torch.ones_like(outputs_A).cuda()

        time_before = time.time()
        outputs_B = model.generate(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            seg_len_tensors=seg_len_tensor,
            max_new_tokens=max_new_tokens,
        )
        time_B = time.time() - time_before

        fw.write(json.dumps({
            "input": example["I_ALL"], "time_A": time_A,
            "output_A": tokenizer.decode(outputs_A[0][input_ids.shape[1]:], skip_special_tokens=True),
            "gt_A": example["O_A"], "time_B": time_B, 
            "output_B": tokenizer.decode(outputs_B[0], skip_special_tokens=True),
            "gt_B": example["O_B"]
        }) + "\n")
        fw.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a pretrained model.")
    parser.add_argument("--model_path", type=str, default="", help="Path to the pretrained model.")
    parser.add_argument("--input_path", type=str, default="data/test.json", help="Path to the input JSON file.")
    parser.add_argument("--seg_order", type=str, nargs='+', default=["I_ALL", "I_A", "O_A"], help="Segment order list.")
    parser.add_argument("--prefix_order", type=str, nargs='+', default=["P_A"], help="Prefix order list.")
    parser.add_argument("--pre_prefix_path", type=str, nargs='+', default=[], help="Path to the previous trained model path.")
    parser.add_argument("--output_path", type=str, default="output/inference.json", help="Path to the output JSONL file.")
    args = parser.parse_args()

    with open(args.output_path, "a") as fw:
        main(args.model_path, args.input_path, args.seg_order, args.prefix_order, args.pre_prefix_path)
