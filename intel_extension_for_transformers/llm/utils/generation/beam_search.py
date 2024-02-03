import torch
from torch import nn
import torch.distributed as dist
import warnings
from typing import Optional, Tuple, Union, List
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.beam_search import BeamScorer
from transformers.utils import ModelOutput
import time
import re


class BeamSearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class BeamSearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]


def _beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    token_latency = (
        self.config.token_latency if hasattr(self.config, "token_latency") else False
    )

    latency_list = []
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn(
            "You don't have defined any stopping_criteria, this will likely loop forever",
            UserWarning,
        )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))
    this_peer_finished = False  # used by synced_gpus only
    while True:
        tic = time.time()
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if (
            re.search("GPTJ", self.config.architectures[0])
            or re.search("llama", self.config.architectures[0], re.IGNORECASE)
            or re.search("gptneox", self.config.architectures[0], re.IGNORECASE)
            or re.search("OPT", self.config.architectures[0], re.IGNORECASE)
            or re.search("falcon", self.config.architectures[0], re.IGNORECASE)
            or re.search("rw", self.config.architectures[0], re.IGNORECASE)
        ):
            first_token = False
            input_bs = input_ids.size()[0]
            has_position_id = True
            if model_inputs["past_key_values"] is None:
                first_token = True
            if first_token and hasattr(self, "trace_graph"):
                if re.search("GPTJ", self.config.architectures[0]):
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long, device=input_ids.device
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long, device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.n_layer)
                        ]
                    )
                elif re.search("llama", self.config.architectures[0], re.IGNORECASE):
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long, device=input_ids.device
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long, device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                elif re.search("gptneox", self.config.architectures[0], re.IGNORECASE):
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long, device=input_ids.device
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long, device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                elif re.search("OPT", self.config.architectures[0], re.IGNORECASE):
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long, device=input_ids.device
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long, device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                    has_position_id = False
                elif re.search(
                    "falcon", self.config.architectures[0], re.IGNORECASE
                ) or re.search("rw", self.config.architectures[0], re.IGNORECASE):
                    beam_idx_tmp = torch.zeros(
                        (2048, int(batch_size * num_beams)), dtype=torch.long, device=input_ids.device
                    ).contiguous()
                    model_inputs["past_key_values"] = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long, device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                torch.zeros([1, 1, 1, 1], device=input_ids.device).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(self.config.num_hidden_layers)
                        ]
                    )
                    has_position_id = False

            if hasattr(self, "trace_graph"):
                if first_token:
                    new_attention_mask = model_inputs["attention_mask"][
                        :batch_size
                    ].clone()
                    new_input_ids = model_inputs["input_ids"][:batch_size].clone()
                    if has_position_id:
                        new_position_ids = model_inputs["position_ids"][
                            :batch_size
                        ].clone()
                    for i in range(batch_size):
                        new_attention_mask[i] = model_inputs["attention_mask"][
                            i * num_beams
                        ]
                        new_input_ids[i] = model_inputs["input_ids"][i * num_beams]
                        if has_position_id:
                            new_position_ids[i] = model_inputs["position_ids"][
                                i * num_beams
                            ]
                    model_inputs["attention_mask"] = new_attention_mask
                    model_inputs["input_ids"] = new_input_ids
                    if has_position_id:
                        model_inputs["position_ids"] = new_position_ids
                model_inputs.pop("use_cache", None)
                model_inputs.pop("token_type_ids", None)
                if first_token and hasattr(self, "trace_graph_first"):
                    outputs = self.trace_graph_first(**model_inputs)
                else:
                    outputs = self.trace_graph(**model_inputs)

                if first_token and len(model_inputs["past_key_values"][1]) == 4:
                    outputs = list(outputs)
                    outputs[0] = outputs[0].repeat_interleave(num_beams, dim=0)
                    outputs = tuple(outputs)
                if synced_gpus and this_peer_finished:
                    cur_len = cur_len + 1
                    continue  # don't waste resources running the code we don't need
                next_token_logits = outputs[0][:, -1, :]
            else:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                if synced_gpus and this_peer_finished:
                    cur_len = cur_len + 1
                    continue  # don't waste resources running the code we don't need
                next_token_logits = outputs.logits[:, -1, :]
        else:
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need
            next_token_logits = outputs.logits[:, -1, :]

        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len
        )
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1
        latency_list.append(time.time() - tic)

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            output_result = BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            output_result = BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        output_result = sequence_outputs["sequences"]
    # result
    if token_latency:
        return (output_result, latency_list)
    else:
        return output_result
