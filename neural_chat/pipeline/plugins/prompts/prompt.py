
from haystack.nodes.prompt.prompt_node import PromptTemplate

def generate_qa_prompt(query, documents=None, history=None, mode="dense"):
    if mode == "dense":
        if documents and history:
            context = " ".join(documents)
            prompt = """Have a conversation with a human, answer the following questions as best you can. 
            You can refer to the following document and context.\n\n### Question: {}\n\n### Context: {}\n\n### Chat History:{}\n\n### Response:""".format(query, context, history)
        elif documents:
            prompt = """Have a conversation with a human, answer the following questions as best you can. 
            You can refer to the following document and context.\n\n### Question: {}\n\n### Context: {}\n\n### Response:""".format(query, context)
        else:
            prompt = """Have a conversation with a human. You are required to generate suitable response to the user input.
                \n\n### Input: {}\n\n### Response:""".format(query)
    else:
        prompt = PromptTemplate(name="llama-instruct", prompt_text="""Have a conversation with a human, answer the following questions as best you can. 
    You can refer to the following document and context.\n### Question: $query\n### Context: $documents\n### Response:""")
    
    return prompt


def generate_prompt(query, history=None):
    if history:
        prompt = """Have a conversation with a human. You are required to generate suitable response to the user input.
                        \n\n### Input:{} \n\n###Chat History: {}\n\n### Response:""".format(query, history)
    else:
        prompt = """Have a conversation with a human. You are required to generate suitable response to the user input.
                        \n\n### Input: {}\n\n### Response:""".format(query)


def generate_intent_prompt(query)
    prompt = """Please identify the intent of the provided context. 
        You may only respond with "chitchat" or "QA" without explanations or engaging in conversation.\nContext:{}\nIntent:""".format(query)
