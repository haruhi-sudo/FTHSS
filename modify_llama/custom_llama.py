import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaModel
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from modify_llama.cascade_attn_mask import make_cascade_mask
from transformers.models.llama.modeling_llama import _expand_mask


class LlamaCascadeAttnmask(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
        self.seg_order = config.seg_order
        self.prefix_order = config.prefix_order
        self.num_virtual_tokens = config.num_virtual_tokens

    def set_segment_order(self, seg_order):
        self.seg_order = seg_order
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        attention_mask, seg_len_tensors = attention_mask[0], attention_mask[1]
        seg_len_dicts = self._get_seg_len_dict(seg_len_tensors)

        cascade_mask_dicts = make_cascade_mask(
            seg_len_dicts,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length
        )

        cascade_masks = []

        for _, cascade_mask_dict in enumerate(cascade_mask_dicts):
            cascade_masks.append(torch.cat(list(cascade_mask_dict.values()), dim=0))

        # combine cascade masks，需要寻找batch中最长的sequence，做padding
        max_size = torch.Size([max(mask.shape[0] for mask in cascade_masks), max(mask.shape[1] for mask in cascade_masks)])
        padded_cascade_masks = [F.pad(mask, (0, max_size[1] - mask.size(1), 0, max_size[0] - mask.size(0)), 
                                      value=0) for mask in cascade_masks]

        combined_attn_mask = torch.stack(padded_cascade_masks)[:,None,:,:]

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            tgt_len = 1 if past_key_values_length > 0 else input_shape[-1]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=tgt_len).to(
                inputs_embeds.device
            )
            combined_attn_mask = (
                expanded_attn_mask if combined_attn_mask is None else expanded_attn_mask + combined_attn_mask
            )

        return combined_attn_mask

    def _get_seg_len_dict(self, seg_len_tensors):
        bsz = seg_len_tensors.shape[0]
        seg_len_dicts = []
        
        for idx in range(bsz):
            seg_len_dict = {}

            for prefix_key in self.prefix_order:
                seg_len_dict[prefix_key] = self.num_virtual_tokens

            for seg_id, seg_key in enumerate(self.seg_order):
                seg_len_dict[seg_key] = seg_len_tensors[idx, seg_id].item()

            seg_len_dicts.append(seg_len_dict)

        return seg_len_dicts

class SpecialLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaCascadeAttnmask(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_segment_order(self, seg_order):
        self.model.set_segment_order(seg_order)

