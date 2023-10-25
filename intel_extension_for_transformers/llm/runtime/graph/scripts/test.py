from functools import partial
from typing import List, Literal, TypedDict, Callable

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def _llama2_format_messages(messages: List[Message], tokenizer_encode: Callable) -> List[int]:
    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]
    assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
        [msg["role"] == "assistant" for msg in messages[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    messages_tokens: List[int] = sum(
        [
            tokenizer_encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                bos=True,
                eos=True,
            )
            for prompt, answer in zip(
                messages[::2],
                messages[1::2],
            )
        ],
        [],
    )
    assert messages[-1]["role"] == "user", f"Last message must be from user, got {messages[-1]['role']}"
    messages_tokens += tokenizer_encode(
        f"{B_INST} {(messages[-1]['content']).strip()} {E_INST}",
        bos=True,
        eos=False,
    )
    return messages_tokens


def _llama_cpp_tokenizer_encode(s: str, bos: bool, eos: bool, llm: Llama) -> List[int]:
    assert type(s) is str
    t = llm.tokenize(text=b" " + bytes(s, encoding="utf-8"), add_bos=False)
    if bos:
        t = [llm.token_bos()] + t
    if eos:
        t = t + [llm.token_eos()]
    return t


class Llama2ChatCompletionWrapper:
    def __init__(self, model_path: str, callback: Callable[[Message], None] = None, tokenizer_encoder: Callable = None) -> None:
        self.llm = Llama(model_path=model_path)
        if tokenizer_encoder is None:
            self._tokenizer_encode = partial(_llama_cpp_tokenizer_encode, llm=self.llm)
        else:
            self._tokenizer_encode = tokenizer_encoder
        self.callback = callback

    def new_session(self, system_content: str | None = None, messages: List[Message] | None = None):
        self.messages: List[Message] = []

        # if self.callback is not None:
        #     self.callback()

        if system_content is not None:
            assert messages is None
            self.messages.append(Message(role="system", content=system_content))
            if self.callback is not None:
                self.callback(self.messages[-1])

        elif messages is not None:
            self.messages = messages
            if self.callback is not None:
                for msg in self.messages:
                    self.callback(msg)

    def __call__(
        self, message: str, post_process: Callable[[str], str] | None = None, max_tokens: int = 128, params: dict = {}
    ) -> str:
        self.messages.append(Message(role="user", content=message))

        if self.callback is not None:
            self.callback(self.messages[-1])

        messages_tokens = _llama2_format_messages(self.messages, tokenizer_encode=self._tokenizer_encode)

        completion = self.llm.generate(messages_tokens, **params)
        max_tokens = (
            max_tokens if max_tokens + len(messages_tokens) < self.llm._n_ctx else (self.llm._n_ctx - len(messages_tokens))
        )
        result = []
        for i, token in enumerate(completion):
            if max_tokens == i or token == self.llm.token_eos():
                break
            result.append(self.llm.detokenize([token]).decode("utf-8"))

        result = "".join(result).strip()

        if post_process is not None:
            # if self.callback is not None:
            #     self.callback()
            result = post_process(result)

        self.messages.append(Message(role="assistant", content=result))
        if self.callback is not None:
            self.callback(self.messages[-1])

        return result