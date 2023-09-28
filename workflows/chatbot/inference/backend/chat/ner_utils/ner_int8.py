import re
import time
import spacy
import datetime
from datetime import timezone, timedelta
from threading import Thread
from transformers import AutoTokenizer, TextStreamer, TextIteratorStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig


model_name = '/home/tme/Llama-2-7b-chat-hf/'
config = WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="int8")
# query = "give me photos of today"
nlp = spacy.load("en_core_web_lg")
month_date_list = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]


def check_query_time(query, cur_time):
    # prompt = """Please determine the precise time mentioned in the user's query. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' If the user query does not include any time reference, please reply with 'None'.
    # \n\n###Current Time:\n{}\n\nUser Query:\n{}\n\nResponse:\n""".format(cur_time, query)
    prompt = """### Instruction: Please thoughtfully identify the precise time range mentioned in the user's query based on the given current time. The response should follows the following requirements. \n
    ### Requirements:
    1. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' 
    2. Please carefully check the accuracy of the identifiction results. 
    3. The phrase "in the last month" means "in the thirty or so days up to and including today".\n
    ### Current Time:\n{}\n
    ### User Query:\n{}\n
    ### Response:\n""".format(cur_time, query)

    return prompt


def enforce_stop_tokens(text: str) -> str:
    """Cut off the text as soon as any stop words occur."""
    stopwords = ["</s"]
    return re.split("|".join(stopwords), text)[0]


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config, trust_remote_code=True)


def inference(query):
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    cur_time = utc_now.astimezone(SHA_TZ).strftime("%Y/%m/%d")
    print("current time is:{}".format(cur_time))
    prompt = check_query_time(query, cur_time)

    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextIteratorStreamer(tokenizer)

    cur_time = time.time()
    result_tokens = model.generate(inputs, streamer=streamer, max_new_tokens=32, seed=1234, threads=56)
    print(f'[ inference time ] {time.time() - cur_time}')
    cur_time = time.time()
    model.model.reinit()

    gen_text = tokenizer.batch_decode(result_tokens)
    result_text = enforce_stop_tokens(gen_text[0])
    print("-------------")
    print(result_text)

    print(f'[ inference time ] {time.time() - cur_time}')
    cur_time = time.time()

    doc = nlp(result_text)
    mentioned_time = {"time":[], "period":[]}
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            if bool(re.search(r'\d', str(ent))):
                print("The target time is {}".format(ent))
                if "to" in result_text:
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
    if len(mentioned_time["period"]) % 2 != 0:
        mentioned_time["time"] = list(set(mentioned_time["time"]+mentioned_time["period"]))
        mentioned_time["period"] = []

    print(f'[ time process time ] {time.time() - cur_time}')
    cur_time = time.time()

    new_doc = nlp(query)
    location = []
    name = []
    organization = []
    s_time = []
    for ent in new_doc.ents:
        if (ent.label_ == 'GPE'):
            location.append(ent.text)
        elif (ent.label_ == 'LOC'):
            location.append(ent.text)
        elif (ent.label_ == 'PERSON'):
            name.append(ent)
        elif (ent.label_ == 'ORG'):
            organization.append(ent)
        elif (ent.label_ == 'DATE' or ent.label_ == 'TIME'):
            s_time.append(ent)
    if s_time == []:
        mentioned_time = {"time": [], "period": []}
    location = list(set(location))

    print(f'[ location process time ] {time.time() - cur_time}')
    cur_time = time.time()

    result_period = []
    for sub in range(len(mentioned_time['period'])//2):
        from_time = mentioned_time['period'][2*sub]
        to_time = mentioned_time['period'][2*sub+1]
        result_period.append({"from": from_time, "to": to_time})
    if 'last month' in query:
        to_time = datetime.datetime.today()
        now_month = to_time.month
        from_time = to_time - timedelta(days=month_date_list[now_month-1])
        result_period = [{"from": str(from_time)[:10], "to": str(to_time)[:10]}]
    result = {"period": result_period, "time": mentioned_time['time'], 'location': location, "name": name, "organization": organization}

    print(f'[ post process time ] {time.time() - cur_time}')
    cur_time = time.time()

    return result


if __name__ == "__main__":
    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            print('exit')
            break
        start_time = time.time()
        result = inference(query)
        print(result)
        end_time = time.time()
        print("Inference cost {} seconds.".format(end_time - start_time))