import json
import torch
import argparse
from peft import PeftConfig
from transformers import AutoTokenizer, AutoConfig
from modify_llama.custom_llama import SpecialLlamaForCausalLM
from peft_config.custom_peft import MyPeftModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnSpecialToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -1] in self.stop_token_id:
            return True
        return False

def main(model_path, input_path, seg_order, prefix_order, init_seg_order = ['I_ALL', 'O_1A'], max_new_tokens=1024, device="cuda"):
    peft_config = PeftConfig.from_pretrained(model_path)
    llama_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)

    llama_config.seg_order = seg_order
    llama_config.prefix_order = prefix_order
    llama_config.num_virtual_tokens = peft_config.num_virtual_tokens // 2

    model = SpecialLlamaForCausalLM.from_pretrained(peft_config.base_model_name_or_path, config=llama_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = MyPeftModelForCausalLM.from_pretrained(
        model, model_path, 
        seg_order=seg_order, 
        prefix_order=prefix_order,
    ).to(torch.bfloat16).to(torch.device(device))

    with open(input_path) as f:
        data = json.load(f)

    reason_start_token = tokenizer.convert_tokens_to_ids("<reasoning>")
    reason_end_token = tokenizer.convert_tokens_to_ids("</reasoning>")
    memory_start_token = tokenizer.convert_tokens_to_ids("<memory>")
    memory_end_token = tokenizer.convert_tokens_to_ids("</memory>")
    stopping_criteria = StoppingCriteriaList([StopOnSpecialToken(stop_token_id=[memory_end_token,reason_end_token])])

    for example in data:
        inputs = tokenizer(example["question"]+"\n", return_tensors="pt", add_special_tokens=False)
        inputs = {name: tensor.to(torch.device(device)) for name, tensor in inputs.items()}
        input_ids, _ = inputs["input_ids"], inputs["attention_mask"]

        seg_len_tensor = torch.tensor([[input_ids.shape[1], 0]]).to(torch.device(device))

        seg_order_round = init_seg_order.copy()
        model.set_segment_order(init_seg_order)
        model.empty_cache()
        
        input_ids_cycle = torch.cat((input_ids, torch.tensor([[reason_start_token]]).to(input_ids)), dim=-1)
        attention_mask_cycle = torch.ones_like(input_ids_cycle).to(input_ids_cycle)

        save_dict = {"I_ALL": example["question"]}
        generation_round = 0
        while True:
            outputs_A = model.generate(
                input_ids=input_ids_cycle,
                attention_mask=attention_mask_cycle,
                seg_len_tensors=seg_len_tensor,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
            generation_text = tokenizer.decode(outputs_A[0,input_ids_cycle.shape[1]:], skip_special_tokens=True)

            # 初始时seg_order_round = ['I_ALL', 'O_1A']，故+1
            save_dict[seg_order[generation_round+1]] = generation_text
            generation_round += 1

            if "[The answer is]" in generation_text or generation_round >= len(seg_order)-1:
                save_dict["gt_answer"] = example["answer"]
                fw.write(json.dumps(save_dict) + "\n")
                fw.flush()
                break

            seg_order_round.append(seg_order[generation_round+1])
            model.set_segment_order(seg_order_round)

            # 准备下一轮的输入
            if generation_round % 2 == 1: # 奇数轮
                input_ids_cycle = torch.tensor([[memory_start_token]]).to(outputs_A)
            else:
                input_ids_cycle = torch.tensor([[reason_start_token]]).to(outputs_A)
            
            if generation_round != 1:
                all_gen_len += outputs_A.shape[1] - 1 #减1的原因是eos token并没有作为下一轮的输入
            else:
                all_gen_len = outputs_A.shape[1]
            
            seg_len_tensor = torch.cat((seg_len_tensor, torch.zeros(1, 1).to(seg_len_tensor)), dim=1)
            attention_mask_cycle = torch.ones([1, all_gen_len]).to(attention_mask_cycle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a pretrained model.")
    parser.add_argument("--model_path", type=str, default="", help="Path to the pretrained model.")
    parser.add_argument("--input_path", type=str, default="data/test.json", help="Path to the input JSON file.")
    parser.add_argument("--seg_order", type=str, nargs='+', default=["I_ALL", "I_A", "O_A"], help="Segment order list.")
    parser.add_argument("--prefix_order", type=str, nargs='+', default=["P_A"], help="Prefix order list.")
    parser.add_argument("--output_path", type=str, default="output.json", help="Path to the output JSONL file.")
    args = parser.parse_args()

    args.seg_order = ["I_ALL", "O_1A", "O_1B", "O_2A", "O_2B", "O_3A", "O_3B", "O_4A", "O_4B", "O_5A", "O_5B"]
    args.prefix_order = ["P_A", "P_B"]

    with open(args.output_path, "a") as fw:
        main(args.model_path, args.input_path, args.seg_order, args.prefix_order)

