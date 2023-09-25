import datetime
from datetime import timezone, timedelta
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

nlp = spacy.load("en_core_web_lg")

model_name ="/home/ubuntu/Llama-2-7b-chat-hf/"
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


def enforce_stop_tokens(text: str) -> str:
    """Cut off the text as soon as any stop words occur."""
    stopwords = ["</s"]
    return re.split("|".join(stopwords), text)[0]


def inference(query):
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    cur_time = utc_now.astimezone(SHA_TZ).strftime("%Y/%m/%d")
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
    text = enforce_stop_tokens(text)
    doc = nlp(text)
    mentioned_time = {"time":[], "period":[]}
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            if bool(re.search(r'\d', str(ent))):
                print("The target time is {}".format(ent))
                if "to" in text:
                    print(f"text: {text}")
                    print(f"ent: {ent}")
                    if "to" in ent.text:
                        cur_periods = ent.text.split(" to ")
                        mentioned_time['period'].extend(cur_periods)
                    else:
                        if len(mentioned_time["period"]) > 0 and mentioned_time["period"][-1] == ent.text:
                            mentioned_time["period"].pop()
                        else:
                            mentioned_time["period"].append(ent.text)
                else:
                    mentioned_time["time"].append(ent.text)
    print("mentioned_time: ", mentioned_time)
    print(len(mentioned_time["period"]))
    if len(mentioned_time["period"]) % 2 != 0:
        mentioned_time["time"] = list(set(mentioned_time["time"]+mentioned_time["period"]))
        mentioned_time["period"] = []

    new_doc = nlp(query)
    location = []
    name = []
    organization = []
    for ent in new_doc.ents:
        if (ent.label_ == 'GPE'):
            location.append(ent.text)
        elif (ent.label_ == 'LOC'):
            location.append(ent.text)
        elif (ent.label_ == 'PERSON'):
            name.append(ent)
        elif (ent.label_ == 'ORG'):
            organization.append(ent)

    location = list(set(location))

    result_period = []
    for sub in range(len(mentioned_time['period'])//2):
        from_time = mentioned_time['period'][2*sub]
        to_time = mentioned_time['period'][2*sub+1]
        result_period.append({"from": from_time, "to": to_time})
    result = {"period": result_period, "time": mentioned_time['time'], 'location': location, "name": name, "organization": organization}

    return result

# CUDA_VISIBLE_DEVICES=0 python test.py
if __name__ == "__main__":

    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            print('exit')
            break
        start_time = time.time()
        result = inference(query)
        print(result)
        # time_period, time_point, locations, names, organizations = result['period'], result['time'], result['location'], result['name'], result['organization']
        # if time_period:
        #     for sub in range(len(time_period)//2):
        #         print("The target time period of query: {} is from {} to {}.".format(query, time_period[2*sub], time_period[2*sub+1]))
        # elif time_point:
        #     for sub in range(len(time_point)):
        #         print("The target time point of query: {} is {}.".format(query, time_point[sub]))
        # if locations:
        #     for loc in locations:
        #         print("The target location of query: {} is {}.".format(query, loc))
        # if names:
        #     for na in names:
        #         print("The mentioned name in the query: {} is {}.".format(query, na))
        # if organizations:
        #     for org in organizations:
        #         print("The mentioned organization in the query: {} is {}.".format(query, org))

        end_time = time.time()
        print("Inference cost {} seconds.".format(end_time - start_time))