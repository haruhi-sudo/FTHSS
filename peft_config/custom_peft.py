import os
import warnings
from typing import Any, Optional, Union, Dict
import torch
import warnings
from transformers.generation.utils import ModelOutput
from peft import PeftModel, PeftType, PeftConfig
from peft.utils import _get_batch_size, load_peft_weights
from typing import Optional


class MyPeftModelForCausalLM(PeftModel):
    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, seg_order: list, 
        prefix_order = None, pre_prefix_path = None, 
        adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

        # seg_order = ["I_ALL", "I_A", "O_A", "O_B"], prefix_order = ["P_A", "P_B"]
        self.seg_order = seg_order
        self.prefix_order = prefix_order if prefix_order is not None else ["P_A"]
        pre_prefix_path = pre_prefix_path if pre_prefix_path is not None else []

        self.pre_prefix_weights = self._get_pre_weights(pre_prefix_path)
        
        self.seg_len_tensors = None
        self.past_key_values = None
    
    # 获取前置模型的已训练参数。注：参数已经被自动冻结
    def _get_pre_weights(self, model_ids):
        adapters_weights = []
        for model_id in model_ids[::-1]:
            adapters_weights.append(load_peft_weights(model_id))
        return adapters_weights
    
    def empty_cache(self):
        self.past_key_values = None

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        seg_order: list,
        prefix_order: list,
        pre_prefix_path = None,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> PeftModel:
        from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")
        
        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable


        model = cls(
            model,
            config,
            seg_order,
            prefix_order,
            pre_prefix_path,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        seg_len_tensors=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, (len(self.pre_prefix_weights)+1)*peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": (attention_mask, seg_len_tensors),
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

            for pre_prefix_weight in self.pre_prefix_weights:
                pre_prompts = pre_prefix_weight["prompt_embeddings"].repeat(batch_size, 1, 1)
                pre_prompts = pre_prompts.to(inputs_embeds.dtype).to(inputs_embeds.device)
                inputs_embeds = torch.cat((pre_prompts, inputs_embeds), dim=1)
            
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._update_model_kwargs_for_generation = self._update_model_kwargs_for_generation
        self.seg_len_tensors = kwargs.get("seg_len_tensors", None)
        
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(*args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(*args, **kwargs)
            else:
                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs
    
    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, seg_len_tensors: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        if peft_config.is_prompt_learning:
            if (model_kwargs["past_key_values"] is None):
                model_kwargs["past_key_values"] = self.past_key_values

            if model_kwargs.get("attention_mask", None) is not None:
                size = model_kwargs["input_ids"].shape[0], (len(self.pre_prefix_weights)+1)*peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                attention_mask = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )
    
                # TODO bsz >= 2
                if self.seg_len_tensors is not None:
                    self.seg_len_tensors[-1,-1] += 1

                model_kwargs["attention_mask"] = (attention_mask, self.seg_len_tensors)
        
            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs["past_key_values"] is None) or (
                not model_kwargs["past_key_values"]
            )

            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                new_past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

                for pre_prefix_weight in self.pre_prefix_weights:
                    pre_prompts = pre_prefix_weight["prompt_embeddings"].repeat(model_kwargs["input_ids"].shape[0], 1, 1)
                    inputs_embeds = torch.cat((pre_prompts, inputs_embeds), dim=1)
            
                model_kwargs["inputs_embeds"] = inputs_embeds
                model_kwargs["input_ids"] = None

        _ = model_kwargs.pop("cache_position", None)

        return model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        if outputs.past_key_values is None:
            model_kwargs["past_key_values"] = self.past_key_values
        else:
            model_kwargs["past_key_values"] = outputs.past_key_values
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        self.past_key_values = model_kwargs["past_key_values"]

        return model_kwargs
