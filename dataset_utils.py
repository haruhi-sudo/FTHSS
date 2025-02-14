from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy
)
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch
import transformers
from typing import Dict
from functools import partial
from transformers import AutoTokenizer


def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int) -> Dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _get_seg_prefix_order(example, output_order="A"):
    seg_order = ["I_ALL"]
    prefix_order = []
    for key, seg_text in example.items():
        if key == "I_ALL":
            continue
        if key.startswith("I_") and key[-1] <= output_order:
            seg_order.append(key)
        elif key.startswith("O_") and key[-1] <= output_order:
            seg_order.append(key)
            prefix_order.append("P_" + key[-1])

    return seg_order, prefix_order

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length=2048, 
                                         output_order="A", prompt_type="llama-3", is_inference=False):
        
    seg_len_dict = {}
    label_mask = []

    example = {key:value.strip() for key, value in example.items()}
    source_text = example["I_ALL"] + "\n"

    source_len = _tokenize_fn(source_text, tokenizer, 1000000)["input_ids_lens"]
    seg_len_dict["I_ALL"] = source_len
    all_text = source_text
    label_mask.append(torch.zeros(source_len)) # 输入不参与计算loss

    for key, seg_text in example.items():
        if key == "I_ALL":
            continue
        if key.startswith("I_") and key[-1] <= output_order:    
            seg_text = seg_text + "\n" # 分隔符
            seg_len = _tokenize_fn(seg_text, tokenizer, 1000000)["input_ids_lens"] # 将分段的max_seq_length设置为一个很大的数
            
            all_text = all_text + seg_text
            label_mask.append(torch.zeros(seg_len)) # 输入不参与计算loss
            seg_len_dict[key] = seg_len

        elif key.startswith("O_") and key[-1] <= output_order: # 例：生成O_B时，只考虑在O_B之前的输出
            if key[-1] == output_order:
                seg_text = tokenizer.bos_token + seg_text + tokenizer.eos_token # 分隔符
                seg_len = _tokenize_fn(seg_text, tokenizer, 1000000)["input_ids_lens"]
                all_text = all_text + seg_text
                
                mask = torch.ones(seg_len+1)
                mask[0] = 0
                label_mask.append(mask) # 输出参与计算mask
                seg_len_dict[key] = seg_len+1
            else:
                seg_text = seg_text + "\n" # 分隔符
                seg_len = _tokenize_fn(seg_text, tokenizer, 1000000)["input_ids_lens"]
                all_text = all_text + seg_text

                label_mask.append(torch.zeros(seg_len))
                seg_len_dict[key] = seg_len
    
    label_mask = torch.cat(label_mask)

    examples_tokenized = _tokenize_fn(all_text, tokenizer, max_seq_length)
    label_mask = label_mask[:max_seq_length]

    # set labels to -100 for the source tokens
    input_ids = examples_tokenized["input_ids"].flatten()
    labels = copy.deepcopy(input_ids)
    labels[label_mask==0] = -100
    attention_mask = torch.ones_like(input_ids)

    if sum(seg_len_dict.values()) != labels.shape[0]:
        labels = torch.full_like(labels, -100)
    
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
        "seg_len_dict": seg_len_dict,
        # "seg_order": list(example.data.keys())
    }

@dataclass
class MyDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    seg_order: list = None

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        seg_len_dicts = []
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                seg_len_dict = feature.pop("seg_len_dict")
                # seg_order = feature.pop("seg_order")
                seg_len_dict = [seg_len_dict[key] for key in self.seg_order]
                seg_len_dicts.append(seg_len_dict)

                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        features["seg_len_tensors"] = torch.tensor(seg_len_dicts, dtype=torch.int64)
        return features

if __name__ == "__main__":
    import json
    from datasets import Dataset, DatasetDict
    data_files = {}
    dataset_args = {}
    data_files["train"] = "data/hotpot_qa/train.json"

    raw_data = {key:json.load(open(path)) for key, path in data_files.items()}

    raw_datasets = DatasetDict({
        key: Dataset.from_list(data) for key, data in raw_data.items()
    })

    seg_order, _ = _get_seg_prefix_order(raw_datasets["train"][0])

    tokenizer = AutoTokenizer.from_pretrained("/data2/LLM_Model/Meta-Llama-3-8B/")
    tokenizer.pad_token = tokenizer.eos_token
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=1024,
        output_order="B",
    )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=1,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True,
        collate_fn=MyDataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", seg_order=seg_order),
        batch_size=2
    )
    for step, batch in enumerate(train_dataloader):
        continue
