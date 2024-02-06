# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import random
import torch
import torch.nn.functional as F

def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None, device="hpu", image_num_patches=576):
    from intel_extension_for_transformers.transformers.modeling.llava_models.llava_arch import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,)

    from .conversation_utils import conv_templates, SeparatorStyle

    def pad_input(input_ids, max_len):
        # fill image placeholder & pre-padding
        padded_input_ids = []
        images_mask = []
        attention_mask = []
        for inp in input_ids:
            new_inp = []
            image_mask = []
            for ele_inp in inp:
                if ele_inp == IMAGE_TOKEN_INDEX:
                    # fill image placeholder with pad token
                    new_inp.extend([IMAGE_TOKEN_INDEX] * image_num_patches)
                    image_mask.extend([1] * image_num_patches)
                else:
                    new_inp.append(ele_inp)
                    image_mask.append(0)


            attn_mask = [1] * len(new_inp)

            if len(new_inp) >= max_len:
                new_inp = new_inp[:max_len]
                image_mask = image_mask[:max_len]
                attn_mask = attn_mask[:max_len]
                p_ids = p_ids[:max_len]
            else:
                # left padding for generation
                inp_len = len(new_inp)
                pad_len = max_len - inp_len
                new_inp = [tokenizer.pad_token_id] * pad_len + new_inp
                image_mask = [0] * pad_len + image_mask
                attn_mask = [0] * pad_len + attn_mask

            assert len(new_inp) == max_len

            image_mask = torch.tensor(image_mask)

            new_inp = torch.tensor(new_inp)[torch.where(image_mask!=1)].tolist()
            if len(new_inp) < max_len:
                inp_len = len(new_inp)
                pad_len = max_len - inp_len
                new_inp = new_inp + [tokenizer.pad_token_id] * pad_len

            padded_input_ids.append(new_inp)
            images_mask.append(image_mask)
            attention_mask.append(attn_mask)

        return torch.tensor(padded_input_ids), torch.stack(images_mask, dim=0), torch.tensor(attention_mask)

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates[args.template].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    if device.type == "hpu":
        # 2048 is max_sequence_length, 128 is max_new_tokens
        input_ids, images_mask, attention_mask = pad_input(input_ids.tolist(), 2048 - 128)

    image = sample['image']
    if image is not None:
        output_ids = model.generate(
            input_ids.to(device),
            images=image.unsqueeze(0).to(torch.bfloat16).to(device),
            attention_mask=attention_mask.to(device),
            images_mask=images_mask.to(device),
            do_sample=False,
            temperature=1,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
            use_cache=True,)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def llava_image_processor(raw_image, vis_processors=None):
    image_tensor = vis_processors.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    return image_tensor
