import torch
from typing import Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from optimum.intel.generation.modeling import TSModelForCausalLM


def jit_trace_mpt_7b(model):
    torch._C._jit_set_texpr_fuser_enabled(False)
    inputs = dict()
    inputs["input_ids"] = torch.ones([4, 2], dtype=torch.int64)
    inputs["attention_mask"] = torch.ones([4, 34], dtype=torch.int64)
    pkv = []
    for i in range(32):
        pkv.append([])
        pkv[-1].append(torch.zeros([4, 32, 128, 32], dtype=torch.bfloat16))
        pkv[-1].append(torch.zeros([4, 32, 32, 128], dtype=torch.bfloat16))
        pkv[-1] = tuple(pkv[-1])
    inputs["past_key_values"] = pkv
    with torch.inference_mode():
        model(*(inputs["input_ids"], inputs["past_key_values"], inputs["attention_mask"]))
        traced_model = torch.jit.trace(model, example_inputs=(inputs["input_ids"], inputs["past_key_values"],
                                       inputs["attention_mask"]), strict=False)
        traced_model = torch.jit.freeze(traced_model.eval())
        traced_model(**inputs)
        traced_model(**inputs)
    return traced_model


class MPTTSModelForCausalLM(TSModelForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_cache:
            if past_key_values is None:
                nb_pkv = 2
                num_layers = self.normalized_config.num_layers
                num_attention_heads = self.normalized_config.num_attention_heads
                hidden_size = self.normalized_config.hidden_size
                d_k = hidden_size // num_attention_heads
                new_key_shape = [input_ids.shape[0], num_attention_heads, d_k, 0]
                new_value_shape = [input_ids.shape[0], num_attention_heads, 0, d_k]
                empty_key_tensor = torch.empty(size=new_key_shape)
                empty_value_tensor = torch.empty(size=new_value_shape)
                if self.model_dtype is not None:
                    empty_key_tensor = empty_key_tensor.to(self.model_dtype)
                    empty_value_tensor = empty_value_tensor.to(self.model_dtype)
                pkv = (empty_key_tensor, empty_value_tensor)
                past_key_values = tuple(tuple(pkv) for _ in range(num_layers))

            inputs["past_key_values"] = past_key_values

        outputs = self.model(**inputs)

        if isinstance(outputs, tuple):
            outputs = CausalLMOutputWithPast(logits=outputs[0], past_key_values=outputs[1] if self.use_cache else None)
        else:
            outputs = CausalLMOutputWithPast(
                logits=outputs["logits"], past_key_values=outputs["past_key_values"] if self.use_cache else None
            )

        return outputs