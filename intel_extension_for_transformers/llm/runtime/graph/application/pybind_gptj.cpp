//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

#define N_threads 56
static model_context** g_ctx;

bool gptj_model_eval_ids(model_context* ctx, model_token* tokens, size_t n_eval, size_t n_past, size_t n_threads) {
  const int n_ctx = model_n_ctx(ctx);
  if (static_cast<int>(n_eval) > n_ctx - 4) {
    fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, static_cast<int>(n_eval),
            n_ctx - 4);
    return 1;
  }

  std::vector<model_input> inputs = {model_input{
      /*.tokens              =*/tokens,
      /*.n_tokens           =*/static_cast<uint32_t>(n_eval),
      /*.n_prompt_tokens    =*/0,
      /*.n_past             =*/static_cast<uint32_t>(n_past),
      /*.n_total            =*/static_cast<uint32_t>(n_past),
      /*.request_idx        =*/0,
      /*.beam_idx           =*/0,
  }};
  if (model_eval(ctx, inputs.data(), inputs.size(), n_threads)) {
    fprintf(stderr, "%s : failed to eval\n", __func__);
    return 1;
  }
  return true;
}

extern "C" {
void* init_gptj(int seed, int n_predict, int n_batch, int top_k, float top_p, float temp, float repeat_penalty,
                bool perplexity, int n_ctx, const char* model_file, bool beam_search = false, int beam_size = 4,
                int batch_size = 1, int n_threads = 56, int min_new_tokens = 0, float length_penalty = 1.0,
                bool do_early_stopping = false) {
  gpt_params params;
  params.n_threads = n_threads;
  params.seed = seed;
  params.model_arch = MODEL_GPTJ;
  params.n_ctx = n_ctx;
  params.n_predict = n_predict;
  params.n_batch = n_batch;
  params.model = std::string(model_file);
  params.n_predict = n_predict;
  params.top_k = top_k;
  params.top_p = top_p;
  params.temp = temp;
  params.repeat_penalty = repeat_penalty;
  params.perplexity = perplexity;
  params.batch_size = batch_size;
  params.beam_search = beam_search;
  params.beam_size = beam_size;
  if (batch_size > 1) params.memory_type = KV_MEM_TYPE_F16;  // TODO(Yi): NO MHA IN MULTI-BATCH
  // params.use_mmap = false;
  // params.use_mlock= true;
  model_init_backend();
  model_context* ctx;
  g_ctx = &ctx;
  ctx = model_init_from_gpt_params(params);
  if (ctx == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return nullptr;
  }
  ctx->generation_conf.min_new_tokens = min_new_tokens;
  ctx->generation_conf.length_penalty = length_penalty;
  ctx->generation_conf.do_early_stopping = do_early_stopping;
  return reinterpret_cast<void*>(ctx);
}

int32_t* eval_gptj_ids(void* ctx, int32_t* embd_inp_ptr, int ind_size, int n_predict, int top_k, float top_p,
                       float temp, int n_batch, int n_threads) {
  model_context* lctx = reinterpret_cast<model_context*>(ctx);
  int n_past = 0;

  auto hparams = lctx->model.hparams;

  n_predict = std::min(n_predict, static_cast<int>(lctx->n_ctx) - static_cast<int>(ind_size));
  std::vector<model_token> res;
  bool do_beam_search = lctx->beam_search;

  if (do_beam_search) {
    std::vector<model_input> inputs = {model_input{
        /*.tokens             =*/embd_inp_ptr,
        /*.n_tokens           =*/static_cast<uint32_t>(ind_size),
        /*.n_prompt_tokens    =*/0,
        /*.n_past             =*/0,
        /*.n_total            =*/0,
        /*.request_idx        =*/0,
        /*.beam_idx           =*/0,
    }};
    res = beam_search(lctx, n_predict, inputs, n_threads)[0];
  } else {
    std::vector<model_token> embd_inp(embd_inp_ptr, embd_inp_ptr + ind_size);
    std::vector<model_token> embd;
    for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
      // predict
      if (embd.size() > 0) {
        if (!gptj_model_eval_ids(lctx, embd.data(), embd.size(), n_past, n_threads)) {
          printf("Failed to predict\n");
          return {};
        }
      }

      auto logits = model_get_logits(lctx);
      n_past += embd.size();
      embd.clear();

      if (i >= embd_inp.size()) {
        const int n_vocab = hparams.n_vocab;
        gpt_vocab::id id = 0;
        id = model_sample_top_k_top_p(lctx, n_vocab, logits, top_k, top_p, temp);
        // add it to the context
        embd.push_back(id);
        res.push_back(id);
      } else {
        // if here, it means we are still processing the input prompt
        for (int k = i; k < embd_inp.size(); k++) {
          embd.push_back(embd_inp[k]);
          if (embd.size() > n_batch) {
            break;
          }
        }
        i += embd.size() - 1;
      }

      // end of text token
      if (embd.back() == 50256) {
        break;
      }
    }
  }
  int32_t* res_ptr = new int32_t[res.size() + 1];
  res_ptr[0] = res.size();
  std::copy(res.begin(), res.end(), &res_ptr[1]);
  return res_ptr;
}

char* eval_gptj_char(void* ctx, const char* prom, int n_predict, int top_k, float top_p, float temp, int n_batch) {
  model_context* lctx = reinterpret_cast<model_context*>(ctx);
  int n_past = 0;

  auto hparams = lctx->model.hparams;
  std::vector<model_token> embd_inp = ::model_tokenize(lctx, std::string(prom), false);
  n_predict = std::min(n_predict, static_cast<int>(lctx->n_ctx) - static_cast<int>(embd_inp.size()));
  std::string res;
  std::vector<model_token> embd;

  bool do_beam_search = lctx->beam_search;
  if (do_beam_search) {
    std::vector<model_input> inputs = {model_input{
        /*.tokens             =*/embd_inp.data(),
        /*.n_tokens           =*/static_cast<uint32_t>(embd_inp.size()),
        /*.n_prompt_tokens    =*/0,
        /*.n_past             =*/0,
        /*.n_total            =*/0,
        /*.request_idx        =*/0,
        /*.beam_idx           =*/0,
        /*.padding_side       =*/0,
        /*n_padding           =*/0,
    }};
    embd = beam_search(lctx, n_predict, inputs, N_threads)[0];
    for (auto id : embd_inp) {
      res += model_token_to_str(lctx, id);
    }
    for (auto id : embd) {
      res += model_token_to_str(lctx, id);
    }
  } else {
    std::vector<float> logits;
    for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
      // predict
      if (embd.size() > 0) {
        if (!gptj_model_eval_ids(lctx, embd.data(), embd.size(), n_past, N_threads)) {
          printf("Failed to predict\n");
          return {};
        }
      }

      auto logits = model_get_logits(lctx);
      n_past += embd.size();
      embd.clear();

      if (i >= embd_inp.size()) {
        const int n_vocab = hparams.n_vocab;
        model_token id = 0;
        id = model_sample_top_k_top_p(lctx, n_vocab, logits, top_k, top_p, temp);
        // add it to the context
        embd.push_back(id);
      } else {
        // if here, it means we are still processing the input prompt
        for (int k = i; k < embd_inp.size(); k++) {
          embd.push_back(embd_inp[k]);
          if (embd.size() > n_batch) {
            break;
          }
        }
        i += embd.size() - 1;
      }
      for (auto id : embd) {
        res += model_token_to_str(lctx, id);
      }

      // end of text token
      if (embd.back() == 50256) {
        break;
      }
    }
  }

  char* res_c_str = new char[res.size() + 1];
  std::strncpy(res_c_str, res.c_str(), res.size());
  return res_c_str;
}

void exit_gptj(void* ctx) {
  model_context* lctx = reinterpret_cast<model_context*>(ctx);
  model_free(lctx);
}
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: ./pybind_gptj <model_filename>\n";
    return 1;
  }

  auto gptj_in_all_bs =
      init_gptj(1234, 32, 32, 40, 1.0, 0.8, 1.02, false, 2048, argv[1], true, 4, 1, 56, 30, 1.0, true);
  std::vector<void*> ctxs = {gptj_in_all_bs};
  for (auto gptj_in_all : ctxs) {
    auto res = eval_gptj_char(
        gptj_in_all,
        // "she opened the door and see",
        // "Once upon a time",
        // "Tell me 10 things about jazz music",
        // "A spaceship lands on the moon",
        // "What is the meaning of life?",
        "2017: It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing "
        "on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. "
        "There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went "
        "right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. "
        "Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of "
        "enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I "
        "could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that "
        "evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt "
        "permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a "
        "species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. "
        "That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve "
        "something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by "
        "creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that "
        "rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to "
        "create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call "
        "it evolution. This is a problem, of course, every other contestant also had to face. And judging by the "
        "entries submitted, not many managed to work around it. I'd say the only real solution was through the use of "
        "artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this "
        "is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed "
        "myself to pick whatever I thought would work out. My initial idea was to create something where humanity "
        "tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had "
        "this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space "
        "Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next "
        "inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are "
        "you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow "
        "gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it "
        "sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey "
        "(who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it "
        "involved into the idea of having individual pieces of pasta flying around and trying to evolve until they "
        "became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti "
        "Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: "
        "you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, "
        "each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through "
        "a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', "
        "which are debited from your credits (you start with a number of credits). Once spawned, your pastas start "
        "flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game "
        "is having your pasta conquer all the plates on the table). But they are really autonomous, so after being "
        "spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other "
        "people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other "
        "pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. "
        "It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If "
        "pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, "
        "until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every "
        "plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, "
        "Carlos' Bistro's founder and owner Setup No major changes were made from my work setup. I used FDT and "
        "Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge "
        "with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing "
        "for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It "
        "made the livestream pretty effortless and the features are awesome, even for the free version. It was great "
        "to have some of my friends watch me, and then interact with them and random people through chat. It was also "
        "good knowing that I was also recording a local version of the files, so I could make a timelapse video later. "
        "Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if "
        "someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly "
        "inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted - it'll "
        "probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty "
        "much spend half of the time writing a line and the other half fixing the crazy characters in it. My own "
        "stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the "
        "cool things to do as a spectator too. It was great seeing other people working - I had a few tabs opened on "
        "my second monitor all the time. It's actually a bit sad, because if I could, I could have spent the whole "
        "weekend just watching other people working! But I had to do my own work, so I'd only do it once in a while, "
        "when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended "
        "up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but "
        "I also went overboard. For example: to know the state of a plate (who owns it, who's conquering it and how "
        "much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at "
        "the plate's bill. The problem I realized when doing some tests is that people never look at the bill! They "
        "think it's some kind of prop, so they never actually read its details. Plus, if you're zoomed out too much, "
        "you can't actually read it, so it's hard to know what's going on with the game until you zoom in to the area "
        "of a specific plate. One other solution that didn't turn out to be as perfect as I thought was how to "
        "indicate who a plate base belongs to. In the game, that's indicated by the plate's decoration - its color "
        "denotes the team owner. But it's something that fits so well into the design that people never realized it, "
        "until they were told about it. In the end, the idea of going with a full physical metaphor is one that should "
        "be done with care. Things that are very important risk becoming background noise, unless the player knows its "
        "importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up "
        "adding it at the bottom to indicate your credits and bases owned, as well as the hideous "
        "out-of-place-and-still-not-obvious 'Call Waiter' button. But in hindsight, I should have gone with a simple "
        "HUD from the start, especially one that indicated each team's colors and general state of the game without "
        "the need for zooming in and out. Development Development went fast.",
        128, 40, 1.0, 0.8, 2048);
    std::cout << res << std::endl;
    exit_gptj(gptj_in_all);
    delete[] res;
    // delete[] res_ids;
  }
  return 0;
}
