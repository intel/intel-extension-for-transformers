"""Function to check the intent of the input user query with LLM."""
import transformers
import torch
from .prompts.prompt import generate_intent_prompt

def intent_detection(model, query, tokenizer):
    """Using the LLM to detect the intent of the user query."""
    prompt = generate_intent_prompt(query)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    generate_ids = model.generate(input_ids, max_new_tokens=5, top_k=1, temperature=0.001)
    intent = tokenizer.batch_decode(generate_ids[:, input_ids.shape[1]:], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)[0]
    return intent