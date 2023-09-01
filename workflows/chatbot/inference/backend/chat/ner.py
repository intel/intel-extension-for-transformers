import datetime
from threading import Thread
import re
import torch
import spacy
import time
# import wikipedia
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    AutoConfig,
)
# CUDA_VISIBLE_DEVICES=6 python test.py

nlp = spacy.load("en_core_web_md")

model_name ="/models/llama-v2-latest-20230719/models_hf_chat/Llama-2-13b-chat-hf/"
print(f"Starting to load the model {model_name} into memory")

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.init_device = 'cuda:0' if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    config=config,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(model_name)
print(f"Successfully loaded the model {model_name} into memory")


def check_query_time(query, cur_time):
    prompt = """Please determine the precise time mentioned in the user's query. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' If the user query does not include any time reference, please reply with 'None'.
    \n\n###Current Time:\n{}\n\nUser Query:\n{}\n\nResponse:\n""".format(cur_time, query)

    return prompt


def inference(query):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d")
    print("current time is:{}".format(cur_time))
    prompt = check_query_time(query, cur_time)
    inputs= tok(prompt, return_token_type_ids=False, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)

    generate_kwargs = dict(
        inputs,
        max_new_tokens=32,
        temperature=0.4,
        top_k=1,
        repetition_penalty=1.1,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    text = ""
    for new_text in streamer:
        text += new_text

    doc = nlp(text)
    mentioned_time = {"time":[], "period":[]}
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            # import pdb;pdb.set_trace()
            if bool(re.search(r'\d', str(ent))):
                print("The target time is {}".format(ent))
                if "to" in text:
                    mentioned_time["period"].append(ent)
                else:
                    mentioned_time["time"].append(ent)
    if len(mentioned_time["period"]) % 2 != 0:
        mentioned_time["time"] = list(set(mentioned_time["time"]+mentioned_time["period"]))
        mentioned_time["period"] = []
    # if mentioned_time["period"] is not None:
    #     import pdb;pdb.set_trace()
    #     datetime_objects = [datetime.strptime(date, '%Y-%m-%d') for date in mentioned_time["period"]]
    #     sorted_dates = sorted(datetime_objects)
    #     mentioned_time["period"] = [postime.strftime("%Y-%m-%d") for postime in sorted_dates]

    new_doc = nlp(query)
    # location = {'GPE':[], 'LOC':[]}
    location = []

    for ent in new_doc.ents:
        if (ent.label_ == 'GPE'):
            # location['GPE'].append(ent.text)
            location.append(ent.text)
        elif (ent.label_ == 'LOC'):
            # location['LOC'].append(ent.text)
            location.append(ent.text)
    location = list(set(location))
    # import pdb;pdb.set_trace()
    # print("The target location is {}".format(location['GPE'][0]))

    return mentioned_time, location


# CUDA_VISIBLE_DEVICES=0 python test.py
if __name__ == "__main__":

    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            print('exit')
            break
        start_time = time.time()
        target_time, location = inference(query)
        if target_time["period"]:
            for sub in range(len(target_time["period"])//2):
                print("The target time period of query: {} is from {} to {}.".format(query, target_time["period"][2*sub], target_time["period"][2*sub+1]))
        else:
            for sub in range(len(target_time["time"])):
                print("The target time period of query: {} is {}.".format(query, target_time["time"][sub]))
        if location:
            for loc in location:
                print("The target location of query: {} is {}.".format(query, loc))

        end_time = time.time()
        print("Inference cost {} seconds.".format(end_time - start_time))
