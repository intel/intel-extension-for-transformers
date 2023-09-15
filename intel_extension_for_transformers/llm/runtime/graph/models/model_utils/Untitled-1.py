model_name = "xxxxx/HF/PATH"
model_name2 = "xxxxx/HF/PATH/2"

import itrex_llm_runtime
m1 = itrex_llm_runtime.Model()
m1.init(model_name, compute_type="int8", n_predict = 20)

while not m1.is_token_end():
    token = m1.gernerate(prompt = prompt)
    print(token, end="", fluse=True)


sentence = m1.gernerate(prompt = prompt)


m2 = itrex_llm_runtime.Model()
m2.init(model_name2, compute_type="int8", n_predict = 20)

sentence = m2.gernerate(prompt = prompt)
