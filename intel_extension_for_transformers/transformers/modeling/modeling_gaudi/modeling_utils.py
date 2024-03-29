# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import transformers

from .generation import (
    GaudiGenerationConfig,
    GaudiGenerationMixin,
    gaudi_MaxLengthCriteria_call,
    gaudi_MaxNewTokensCriteria_call,
)
from .models import (
    GaudiBloomForCausalLM,
    GaudiBloomMLP,
    GaudiCodeGenAttention,
    GaudiCodeGenForCausalLM,
    GaudiFalconForCausalLM,
    GaudiFalconModel,
    GaudiGPT2Attention,
    GaudiGPT2LMHeadModel,
    GaudiGPTBigCodeForCausalLM,
    GaudiGPTJAttention,
    GaudiGPTJForCausalLM,
    GaudiGPTNeoXForCausalLM,
    GaudiLlamaAttention,
    GaudiLlamaDecoderLayer,
    GaudiLlamaDynamicNTKScalingRotaryEmbedding,
    GaudiLlamaForCausalLM,
    GaudiLlamaLinearScalingRotaryEmbedding,
    GaudiLlamaMLP,
    GaudiLlamaModel,
    GaudiLlamaRotaryEmbedding,
    GaudiMistralAttention,
    GaudiMistralDecoderLayer,
    GaudiMistralForCausalLM,
    GaudiMistralModel,
    GaudiMixtralForCausalLM,
    GaudiMptForCausalLM,
    GaudiMptModel,
    GaudiOPTForCausalLM,
    GaudiOPTLearnedPositionalEmbedding,
    GaudiPhiForCausalLM,
    _gaudi_wav2vec2_compute_mask_indices,
    _gaudi_wav2vec2_mask_hidden_states,
    gaudi_albert_forward,
    gaudi_BartAttention_forward,
    gaudi_BartDecoder_forward,
    gaudi_BartDecoderLayer_forward,
    gaudi_BartEncoder_forward,
    gaudi_BartEncoderLayer_forward,
    gaudi_BartForConditionalGeneration_forward,
    gaudi_BartForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_BartLearnedPositionalEmbedding,
    gaudi_BartModel_forward,
    gaudi_BlipForConditionalGeneration_generate,
    gaudi_BlipForQuestionAnswering_generate,
    gaudi_BlipTextAttention_forward,
    gaudi_BlipTextEncoder_forward,
    gaudi_BlipTextLayer_forward,
    gaudi_BlipTextLMHead_forward,
    gaudi_BlipTextLMHead_prepare_inputs_for_generation,
    gaudi_BlipTextModel_forward,
    gaudi_BlipTextSelfAttention_forward,
    gaudi_bloom_attention_forward,
    gaudi_bloom_block_forward,
    gaudi_bloom_convert_to_bloom_cache,
    gaudi_bloom_convert_to_standard_cache,
    gaudi_bloom_model_forward,
    gaudi_check_and_enable_sdpa,
    gaudi_codegen_block_forward,
    gaudi_codegen_model_forward,
    gaudi_conv1d_forward,
    gaudi_esm_for_protein_folding_forward,
    gaudi_esmfolding_trunk_forward,
    gaudi_falcon_attention_forward,
    gaudi_falcon_attention_split_heads,
    gaudi_falcon_decoder_layer_forward,
    gaudi_generate_speech,
    gaudi_get_extended_attention_mask,
    gaudi_gpt2_block_forward,
    gaudi_gpt2_forward,
    gaudi_gpt_bigcode_attention_forward,
    gaudi_gpt_bigcode_block_forward,
    gaudi_gpt_bigcode_model_forward,
    gaudi_gpt_neox_attention_forward,
    gaudi_gpt_neox_layer_forward,
    gaudi_gpt_neox_model_forward,
    gaudi_gpt_neox_rotary_embedding_set_cos_sin_cache,
    gaudi_gptj_block_forward,
    gaudi_gptj_model_forward,
    gaudi_invert_attention_mask,
    gaudi_llama_rmsnorm_forward,
    gaudi_mistral_rmsnorm_forward,
    gaudi_mixtral_attention_forward,
    gaudi_mixtral_block_sparse_moe_forward,
    gaudi_mixtral_decoder_layer_forward,
    gaudi_mixtral_model_forward,
    gaudi_mixtral_rmsnorm_forward,
    gaudi_mpt_attention_forward,
    gaudi_mpt_block_forward,
    gaudi_opt_attention_forward,
    gaudi_opt_decoder_forward,
    gaudi_opt_decoder_layer_forward,
    gaudi_opt_model_forward,
    gaudi_phi_attention_forward,
    gaudi_phi_decoder_layer_forward,
    gaudi_phi_model_forward,
    gaudi_rot_matmul,
    gaudi_rot_vec_mul,
    gaudi_SpeechT5Attention_forward,
    gaudi_SpeechT5Decoder_forward,
    gaudi_SpeechT5DecoderLayer_forward,
    gaudi_SpeechT5SpeechDecoderPrenet_forward,
    gaudi_swin_get_attn_mask,
    gaudi_t5_layernorm_forward,
    gaudi_T5Attention_forward,
    gaudi_T5Block_forward,
    gaudi_T5ForConditionalGeneration_forward,
    gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_T5LayerSelfAttention_forward,
    gaudi_T5Stack_forward,
    gaudi_vit_self_attention_forward,
    gaudi_wav2vec2_encoder_forward,
    gaudi_wav2vec2_forward,
    gaudi_wav2vec2_tdnnlayer_forward,
)


def adapt_transformers_to_gaudi():
    """
    Replaces some Transformers' methods for equivalent methods optimized
    for Gaudi.
    """

    # optimize Conv1D
    transformers.pytorch_utils.Conv1D.forward = gaudi_conv1d_forward

    # Optimization tweak for ViT
    transformers.models.vit.modeling_vit.ViTSelfAttention.forward = gaudi_vit_self_attention_forward

    # Optimization tweak for Swin
    transformers.models.swin.modeling_swin.SwinLayer.get_attn_mask = gaudi_swin_get_attn_mask

    # Optimization tweak for Wav2Vec2
    transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices = _gaudi_wav2vec2_compute_mask_indices
    # transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices = _gaudi_wav2vec2_sample_negative_indices
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states = (
        _gaudi_wav2vec2_mask_hidden_states
    )
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.forward = gaudi_wav2vec2_forward
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder.forward = gaudi_wav2vec2_encoder_forward
    transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer.forward = gaudi_wav2vec2_tdnnlayer_forward

    # Generation is modified to run faster in lazy mode
    transformers.generation.GenerationMixin.generate = GaudiGenerationMixin.generate
    transformers.generation.GenerationMixin._update_model_kwargs_for_generation = (
        GaudiGenerationMixin._update_model_kwargs_for_generation
    )
    transformers.generation.GenerationMixin.update_model_kwargs_for_bucketing = (
        GaudiGenerationMixin.update_model_kwargs_for_bucketing
    )
    transformers.generation.GenerationMixin._get_hpu_graphs_kwargs = GaudiGenerationMixin._get_hpu_graphs_kwargs
    transformers.generation.GenerationMixin._expand_inputs_for_generation = staticmethod(
        GaudiGenerationMixin._expand_inputs_for_generation
    )
    transformers.generation.GenerationMixin._prepare_attention_mask_for_generation = (
        GaudiGenerationMixin._prepare_attention_mask_for_generation
    )
    transformers.generation.GenerationMixin._prepare_decoder_input_ids_for_generation = (
        GaudiGenerationMixin._prepare_decoder_input_ids_for_generation
    )
    transformers.generation.GenerationMixin._prepare_decoder_attention_mask = (
        GaudiGenerationMixin._prepare_decoder_attention_mask
    )
    transformers.generation.GenerationMixin._validate_model_kwargs = GaudiGenerationMixin._validate_model_kwargs
    transformers.generation.GenerationMixin.greedy_search = GaudiGenerationMixin.greedy_search
    transformers.generation.GenerationMixin.sample = GaudiGenerationMixin.sample
    transformers.generation.GenerationMixin.beam_search = GaudiGenerationMixin.beam_search
    transformers.generation.GenerationMixin.beam_sample = GaudiGenerationMixin.beam_sample
    transformers.generation.GenerationMixin.group_beam_search = GaudiGenerationMixin.group_beam_search
    transformers.generation.GenerationMixin.constrained_beam_search = GaudiGenerationMixin.constrained_beam_search
    transformers.generation.GenerationConfig = GaudiGenerationConfig
    transformers.modeling_utils.GenerationConfig = GaudiGenerationConfig
    transformers.generation.MaxLengthCriteria.__call__ = gaudi_MaxLengthCriteria_call
    transformers.generation.MaxNewTokensCriteria.__call__ = gaudi_MaxNewTokensCriteria_call

    # Optimization for BLOOM generation on Gaudi
    transformers.models.bloom.modeling_bloom.BloomAttention.forward = gaudi_bloom_attention_forward
    transformers.models.bloom.modeling_bloom.BloomBlock.forward = gaudi_bloom_block_forward
    transformers.models.bloom.modeling_bloom.BloomModel.forward = gaudi_bloom_model_forward
    transformers.models.bloom.modeling_bloom.BloomMLP = GaudiBloomMLP
    transformers.models.bloom.modeling_bloom.BloomForCausalLM = GaudiBloomForCausalLM
    transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache = (
        gaudi_bloom_convert_to_standard_cache
    )
    transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache = (
        gaudi_bloom_convert_to_bloom_cache
    )

    # Optimization for BART generation on Gaudi
    transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding = gaudi_BartLearnedPositionalEmbedding
    transformers.models.bart.modeling_bart.BartAttention.forward = gaudi_BartAttention_forward
    transformers.models.bart.modeling_bart.BartEncoderLayer.forward = gaudi_BartEncoderLayer_forward
    transformers.models.bart.modeling_bart.BartDecoderLayer.forward = gaudi_BartDecoderLayer_forward
    transformers.models.bart.modeling_bart.BartEncoder.forward = gaudi_BartEncoder_forward
    transformers.models.bart.modeling_bart.BartDecoder.forward = gaudi_BartDecoder_forward
    transformers.models.bart.modeling_bart.BartModel.forward = gaudi_BartModel_forward
    transformers.models.bart.modeling_bart.BartForConditionalGeneration.forward = (
        gaudi_BartForConditionalGeneration_forward
    )
    transformers.models.bart.modeling_bart.BartForConditionalGeneration.prepare_inputs_for_generation = (
        gaudi_BartForConditionalGeneration_prepare_inputs_for_generation
    )

    # Optimization for codegen generation on Gaudi
    transformers.models.codegen.modeling_codegen.CodeGenAttention = GaudiCodeGenAttention
    transformers.models.codegen.modeling_codegen.CodeGenForCausalLM = GaudiCodeGenForCausalLM
    transformers.models.codegen.modeling_codegen.CodeGenModel.forward = gaudi_codegen_model_forward
    transformers.models.codegen.modeling_codegen.CodeGenBlock.forward = gaudi_codegen_block_forward

    # Replace invert_attention_mask and get_extended_attention_mask
    # so that Torch Autocast is disabled for specific parts of the code
    transformers.modeling_utils.ModuleUtilsMixin.invert_attention_mask = gaudi_invert_attention_mask
    transformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_mask = gaudi_get_extended_attention_mask

    # Override sdpa check on Gaudi
    transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa = gaudi_check_and_enable_sdpa

    # AlbertModel.forward does not rely on get_extended_attention_mask so it also needs to be replaced
    transformers.models.albert.modeling_albert.AlbertModel.forward = gaudi_albert_forward

    # Optimization for GPT2 on Gaudi
    transformers.models.gpt2.modeling_gpt2.GPT2Attention = GaudiGPT2Attention
    transformers.models.gpt2.modeling_gpt2.GPT2Model.forward = gaudi_gpt2_forward
    transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel = GaudiGPT2LMHeadModel
    transformers.models.gpt2.modeling_gpt2.GPT2Block.forward = gaudi_gpt2_block_forward

    # Optimization for EsmFold on Gaudi
    transformers.models.esm.modeling_esmfold.EsmFoldingTrunk.forward = gaudi_esmfolding_trunk_forward
    transformers.models.esm.modeling_esmfold.EsmForProteinFolding.forward = gaudi_esm_for_protein_folding_forward
    transformers.models.esm.openfold_utils.rigid_utils.rot_matmul = gaudi_rot_matmul
    transformers.models.esm.openfold_utils.rigid_utils.rot_vec_mul = gaudi_rot_vec_mul

    # Optimization for OPT generation on Gaudi
    transformers.models.opt.modeling_opt.OPTAttention.forward = gaudi_opt_attention_forward
    transformers.models.opt.modeling_opt.OPTDecoder.forward = gaudi_opt_decoder_forward
    transformers.models.opt.modeling_opt.OPTForCausalLM = GaudiOPTForCausalLM
    transformers.models.opt.modeling_opt.OPTModel.forward = gaudi_opt_model_forward
    transformers.models.opt.modeling_opt.OPTDecoderLayer.forward = gaudi_opt_decoder_layer_forward
    transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding = GaudiOPTLearnedPositionalEmbedding

    # Optimization for GPTJ on Gaudi
    transformers.models.gptj.modeling_gptj.GPTJAttention = GaudiGPTJAttention
    transformers.models.gptj.modeling_gptj.GPTJForCausalLM = GaudiGPTJForCausalLM
    transformers.models.gptj.modeling_gptj.GPTJBlock.forward = gaudi_gptj_block_forward
    transformers.models.gptj.modeling_gptj.GPTJModel.forward = gaudi_gptj_model_forward

    # Optimization for GPTBigCode on Gaudi
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention.forward = (
        gaudi_gpt_bigcode_attention_forward
    )
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM = GaudiGPTBigCodeForCausalLM
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeBlock.forward = gaudi_gpt_bigcode_block_forward
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeModel.forward = gaudi_gpt_bigcode_model_forward

    # Optimization for gpt-neox generation on Gaudi
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM = GaudiGPTNeoXForCausalLM
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXModel.forward = gaudi_gpt_neox_model_forward
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer.forward = gaudi_gpt_neox_layer_forward
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention.forward = gaudi_gpt_neox_attention_forward
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXRotaryEmbedding._set_cos_sin_cache = (
        gaudi_gpt_neox_rotary_embedding_set_cos_sin_cache
    )

    # Optimization for llama generation on Gaudi
    transformers.models.llama.modeling_llama.LlamaForCausalLM = GaudiLlamaForCausalLM
    transformers.models.llama.modeling_llama.LlamaModel = GaudiLlamaModel
    transformers.models.llama.modeling_llama.LlamaAttention = GaudiLlamaAttention
    transformers.models.llama.modeling_llama.LlamaMLP = GaudiLlamaMLP
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = GaudiLlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = GaudiLlamaRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = GaudiLlamaLinearScalingRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = (
        GaudiLlamaDynamicNTKScalingRotaryEmbedding
    )
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = gaudi_llama_rmsnorm_forward

    # Optimization for falcon generation on Gaudi
    transformers.models.falcon.modeling_falcon.FalconForCausalLM = GaudiFalconForCausalLM
    transformers.models.falcon.modeling_falcon.FalconModel = GaudiFalconModel
    transformers.models.falcon.modeling_falcon.FalconDecoderLayer.forward = gaudi_falcon_decoder_layer_forward
    transformers.models.falcon.modeling_falcon.FalconAttention.forward = gaudi_falcon_attention_forward
    transformers.models.falcon.modeling_falcon.FalconAttention._split_heads = gaudi_falcon_attention_split_heads

    # Optimization for t5 on Gaudi
    transformers.models.t5.modeling_t5.T5LayerNorm.forward = gaudi_t5_layernorm_forward
    transformers.models.t5.modeling_t5.T5Stack.forward = gaudi_T5Stack_forward
    transformers.models.t5.modeling_t5.T5LayerSelfAttention.forward = gaudi_T5LayerSelfAttention_forward
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration.forward = gaudi_T5ForConditionalGeneration_forward
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation = (
        gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation
    )
    transformers.models.t5.modeling_t5.T5Attention.forward = gaudi_T5Attention_forward
    transformers.models.t5.modeling_t5.T5Block.forward = gaudi_T5Block_forward

    # Optimization for mpt on Gaudi
    transformers.models.mpt.modeling_mpt.MptForCausalLM = GaudiMptForCausalLM
    transformers.models.mpt.modeling_mpt.MptModel = GaudiMptModel
    transformers.models.mpt.modeling_mpt.MptAttention.forward = gaudi_mpt_attention_forward
    transformers.models.mpt.modeling_mpt.MptBlock.forward = gaudi_mpt_block_forward

    # Optimization for mistral on Gaudi
    transformers.models.mistral.modeling_mistral.MistralForCausalLM = GaudiMistralForCausalLM
    transformers.models.mistral.modeling_mistral.MistralAttention = GaudiMistralAttention
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer = GaudiMistralDecoderLayer
    transformers.models.mistral.modeling_mistral.MistralModel = GaudiMistralModel
    transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = gaudi_mistral_rmsnorm_forward

    # Optimization for phi on Gaudi
    transformers.models.phi.modeling_phi.PhiForCausalLM = GaudiPhiForCausalLM
    transformers.models.phi.modeling_phi.PhiAttention.forward = gaudi_phi_attention_forward
    transformers.models.phi.modeling_phi.PhiDecoderLayer.forward = gaudi_phi_decoder_layer_forward
    transformers.models.phi.modeling_phi.PhiModel.forward = gaudi_phi_model_forward

    # Optimization for blip Text model on Gaudi
    transformers.models.blip.BlipTextModel.forward = gaudi_BlipTextModel_forward
    transformers.models.blip.modeling_blip_text.BlipTextLMHeadModel.forward = gaudi_BlipTextLMHead_forward
    transformers.models.blip.modeling_blip_text.BlipTextLMHeadModel.prepare_inputs_for_generation = (
        gaudi_BlipTextLMHead_prepare_inputs_for_generation
    )
    transformers.models.blip.modeling_blip_text.BlipTextEncoder.forward = gaudi_BlipTextEncoder_forward
    transformers.models.blip.modeling_blip_text.BlipTextLayer.forward = gaudi_BlipTextLayer_forward
    transformers.models.blip.modeling_blip_text.BlipTextAttention.forward = gaudi_BlipTextAttention_forward
    transformers.models.blip.modeling_blip_text.BlipTextSelfAttention.forward = gaudi_BlipTextSelfAttention_forward
    transformers.models.blip.BlipForQuestionAnswering.generate = gaudi_BlipForQuestionAnswering_generate
    transformers.models.blip.BlipForConditionalGeneration.generate = gaudi_BlipForConditionalGeneration_generate

    # Optimization for mixtral on Gaudi
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM = GaudiMixtralForCausalLM
    transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = gaudi_mixtral_model_forward
    transformers.models.mixtral.modeling_mixtral.MixtralAttention.forward = gaudi_mixtral_attention_forward
    transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward = gaudi_mixtral_block_sparse_moe_forward
    transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer.forward = gaudi_mixtral_decoder_layer_forward
    transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm.forward = gaudi_mixtral_rmsnorm_forward

    # Optimization for speecht5 on Gaudi
    transformers.models.speecht5.modeling_speecht5.SpeechT5Decoder.forward = gaudi_SpeechT5Decoder_forward
    transformers.models.speecht5.modeling_speecht5.SpeechT5DecoderLayer.forward = gaudi_SpeechT5DecoderLayer_forward
    transformers.models.speecht5.modeling_speecht5.SpeechT5Attention.forward = gaudi_SpeechT5Attention_forward
    transformers.models.speecht5.modeling_speecht5._generate_speech = gaudi_generate_speech
    transformers.models.speecht5.modeling_speecht5.SpeechT5SpeechDecoderPrenet.forward = (
        gaudi_SpeechT5SpeechDecoderPrenet_forward
    )
