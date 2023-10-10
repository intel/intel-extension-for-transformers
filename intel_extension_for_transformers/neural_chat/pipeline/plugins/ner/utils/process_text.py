import re
import datetime
from datetime import timedelta


month_date_list = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]


def post_process_last_month() -> list[dict]:
    to_time = datetime.datetime.today()
    now_month = to_time.month
    from_time = to_time - timedelta(days=month_date_list[now_month-1])
    result_period = [{"from": str(from_time)[:10], "to": str(to_time)[:10]}]
    return result_period


def post_process_last_week() -> list[dict]:
    to_time = datetime.datetime.today()
    from_time = to_time - timedelta(days=7)
    result_period = [{"from": str(from_time)[:10], "to": str(to_time)[:10]}]
    return result_period


def process_time(result_text: str, doc) -> dict:
    mentioned_time = {"time":[], "period":[]}
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            if bool(re.search(r'\d', str(ent))):
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
    if len(mentioned_time["period"]) % 2 != 0:
        mentioned_time["time"] = list(set(mentioned_time["time"]+mentioned_time["period"]))
        mentioned_time["period"] = []
    
    return mentioned_time


def process_entities(query, doc, mentioned_time: dict) -> dict:
    location = []
    name = []
    organization = []
    s_time = []
    for ent in doc.ents:
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

    result_period = []
    for sub in range(len(mentioned_time['period'])//2):
        from_time = mentioned_time['period'][2*sub]
        to_time = mentioned_time['period'][2*sub+1]
        result_period.append({"from": from_time, "to": to_time})

    # post process
    if 'last month' in query:
        result_period = post_process_last_month()
    if 'last week' in query:
        result_period = post_process_last_week()

    result = {"period": result_period, "time": mentioned_time['time'], 'location': location, "name": name, "organization": organization}
    
    return result