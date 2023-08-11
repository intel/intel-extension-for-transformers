"""Function to check the intent of the input user query with LLM."""
import transformers
import torch

def intent_detection(model, query, tokenizer):
    """Using the LLM to detect the intent of the user query."""
    prompt = """Please identify the intent of the provided context. 
        You may only respond with "chitchat" or "QA" without explanations or engaging in conversation.\nContext:{}\nIntent:""".format(
        query)
    input_ids = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=2043, truncation=True, ).input_ids
    input_ids = input_ids.to(model.device)
    generate_ids = model.generate(input_ids, max_new_tokens=5, top_k=3)
    intent = tokenizer.batch_decode(generate_ids[:, input_ids.shape[1]:], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)[0]
    return intent