#  Copyright (c) 2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import time
import os
from vllm import LLM, SamplingParams
from typing import List, Optional
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, RtnConfig
from transformers import AutoTokenizer

prompt_32 = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
prompt_64 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot of things I would like to talk about.",
prompt_128 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, instead of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors",
prompt_256 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, instead of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective",
prompt_512 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, instead of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dilemma when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby"
prompt_1024 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, instead of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dilemma when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate,"

prompts = [prompt_32, prompt_1024]


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model name: String", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--use_neural_speed", action="store_true")
    args = parser.parse_args(args_in)
    print(args)

    if args.benchmark:
        sampling_params = SamplingParams(max_tokens=32)

        config = RtnConfig(compute_dtype="bf16",
                           group_size=128,
                           scale_dtype="bf16",
                           weight_dtype="int4_clip",
                           bits=4)
        llm = LLM(model=args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True, config=config)

        for prompt in prompts:
            vllm_outputs = llm.generate(prompt, sampling_params)  # Generate texts from the prompts.
            qbits_output = model.generate(prompt, sampling_params)

            print("vLLM outputs = ", vllm_outputs)
            print("vLLM input_tokens_length = ", len(vllm_outputs[0].prompt_token_ids),
                  "output_tokens_length = ", len(vllm_outputs[0].outputs[0].token_ids))
            print('The vLLM generate = ',
                  vllm_outputs[0].metrics.finished_time - vllm_outputs[0].metrics.arrival_time, "s")
            print("The vLLM first token time = ",
                  vllm_outputs[0].metrics.first_token_time - vllm_outputs[0].metrics.first_scheduled_time)

            print("QBits_vLLM outputs = ", qbits_output)
            print("QBits_vLLM input_tokens_length = ", len(qbits_output[0].prompt_token_ids),
                  "output_tokens_length = ", len(qbits_output[0].outputs[0].token_ids))
            print('The QBits optimized generate = ',
                  qbits_output[0].metrics.finished_time - qbits_output[0].metrics.arrival_time, "s")
            print("The QBits first token time = ",
                  qbits_output[0].metrics.first_token_time - qbits_output[0].metrics.first_scheduled_time)

            if args.use_neural_speed:
                os.environ["NEURAL_SPEED_VERBOSE"] = "1"
                woq_config = RtnConfig(bits=4, weight_dtype="int4", compute_dtype="int8", scale_dtype="bf16")
                model_with_ns = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                                     quantization_config=woq_config)

                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                inputs = tokenizer(args.prompt, return_tensors="pt").input_ids

                output = model_with_ns.generate(inputs, max_new_tokens=32)
                print("neural speed output = ", output)
        return

    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True)
    output = model.generate(args.prompt)
    print(output)


if __name__ == "__main__":
    main()
