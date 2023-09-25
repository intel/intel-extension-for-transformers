import datetime
from threading import Thread
import re
import torch
import spacy
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    AutoConfig,
)


def check_query_time(query, cur_time):
    prompt = """Please determine the precise time mentioned in the user's query. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' If the user query does not include any time reference, please reply with 'None'.
    \n\n###Current Time:\n{}\n\nUser Query:\n{}\n\nResponse:\n""".format(cur_time, query)

    return prompt


def inference(query, tok, model, nlp):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d")
    print("current time is:{}".format(cur_time))
    prompt = check_query_time(query, cur_time)
    inputs= tok(prompt, return_token_type_ids=False, return_tensors="pt")
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
            if bool(re.search(r'\d', str(ent))):
                print("The target time is {}".format(ent))
                if "to" in text:
                    mentioned_time["period"].append(ent)
                else:
                    mentioned_time["time"].append(ent)
    if len(mentioned_time["period"]) % 2 != 0:
        mentioned_time["time"] = list(set(mentioned_time["time"]+mentioned_time["period"]))
        mentioned_time["period"] = []

    new_doc = nlp(query)
    location = []

    for ent in new_doc.ents:
        if (ent.label_ == 'GPE'):
            location.append(ent.text)
        elif (ent.label_ == 'LOC'):
            location.append(ent.text)
    location = list(set(location))

    return mentioned_time, location


def generate_query_from_prompt(query):
    # load model
    nlp = spacy.load("en_core_web_md")
    model_name ="/home/ubuntu/Llama-2-7b-chat-hf/"
    print(f"Starting to load the model {model_name} into memory")

    # initiate model and config
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

    # inference
    start_time = time.time()
    target_time, location = inference(query, tok, model, nlp)

    # construct results
    result = {}
    if target_time["period"]:
        result['period'] = []
        for sub in range(len(target_time["period"])//2):
            from_time = str(target_time["period"][2*sub]).split('<')[0]
            to_time = str(target_time["period"][2*sub+1]).split('<')[0]
            result['period'].append({"from": from_time, "to": to_time})
    else:
        result['time'] = []
        for sub in range(len(target_time["time"])):
            result['time'].append(str(target_time["time"][sub]).split('<')[0])
    if location:
        result['location'] = []
        for loc in location:
            result['location'].append(loc)
    return result


if __name__ == "__main__":
    query_list = [
        "show me photos at 2023.8.30.",
        "show me photos at 1st August.",
        "Give me photos taken last week at shanghai."
    ]
    for query in query_list:
        print(f'--------- [{query}] ----------')
        result = generate_query_from_prompt(query)
        print(result)
