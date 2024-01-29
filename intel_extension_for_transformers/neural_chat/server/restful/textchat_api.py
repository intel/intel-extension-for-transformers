#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from http import HTTPStatus
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, Response
# pylint: disable=E0611
from pydantic import BaseModel
from typing import Optional, AsyncIterator
from fastapi import APIRouter
from ...cli.log import logger
from ...server.restful.openai_protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from ...server.restful.openai_protocol import ModelCard, ModelList, ModelPermission, ApiErrorCode
from ...config import GenerationConfig
import json, types
from ...plugins import plugins, is_plugin_enabled

def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.top_k is not None and (request.top_k > -1 and request.top_k < 1):
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_k} is out of Range. Either set top_k to -1 or >=1.",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ApiErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None

def create_error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(content=ErrorResponse(message=message, code=status_code),
                        status_code=status_code.value)

async def check_model(request) -> Optional[JSONResponse]:
    if request.model == router.get_chatbot().model_name:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret

class TextChatAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()

    def set_chatbot(self, chatbot, use_deepspeed=False, world_size=1, host="0.0.0.0", port=80) -> None:
        self.chatbot = chatbot
        self.use_deepspeed = use_deepspeed
        self.world_size = world_size
        self.host = host
        self.port = port

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot

    def is_generator(self, obj):
        return isinstance(obj, types.GeneratorType)

    def handle_completion_request(self, request: ChatCompletionRequest):
        return self.handle_chat_completion_request(request)


    async def handle_chat_completion_request(self, request: ChatCompletionRequest):
        chatbot = self.get_chatbot()

        try:
            logger.info(f"Predicting chat completion using prompt '{request.prompt}'")
            config = GenerationConfig()
            # Set attributes of the config object from the request
            for attr, value in request.__dict__.items():
                if attr == "stream":
                    continue
                setattr(config, attr, value)
            if chatbot.device == "hpu":
                config.device = "hpu"
                config.use_hpu_graphs = True
                config.task = "chat"
            buffered_texts = ""
            if request.stream:
                generator, link = chatbot.predict_stream(query=request.prompt, config=config)
                if not self.is_generator(generator):
                    generator = (generator,)
                def stream_generator():
                    nonlocal buffered_texts
                    for output in generator:
                        if isinstance(output, str):
                            chunks = output.split()
                            for chunk in chunks:
                                ret = {
                                    "text": chunk,
                                    "error_code": 0,
                                }
                                buffered_texts += chunk + ' '
                                yield json.dumps(ret).encode() + b"\0"
                        else:
                            ret = {
                                "text": output,
                                "error_code": 0,
                            }
                            buffered_texts += output + ' '
                            yield json.dumps(ret).encode() + b"\0"
                    yield f"data: [DONE]\n\n"
                    if is_plugin_enabled("cache") and \
                       not plugins["cache"]["instance"].pre_llm_inference_actions(request.prompt):
                        plugins["cache"]["instance"].post_llm_inference_actions(request.prompt, buffered_texts)
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                ret = chatbot.predict(query=request.prompt, config=config)
                if isinstance(ret, AsyncIterator):
                    async for request_output in ret:
                        # top 1 request outputs
                        final_output = request_output.outputs[0].text
                        # TODO streaming response
                        # return ChatCompletionResponse(response=final_output)
                    response = final_output
                else:
                    response = ret
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        else:
            logger.info(f"Chat completion finished.")
            return ChatCompletionResponse(response=response)


router = TextChatAPIRouter()


@router.post("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=router.get_chatbot().model_name,
                  root=router.get_chatbot().model_name,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)

@router.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message.
    This API mimics the OpenAI ChatCompletion API.

    See  https://platform.openai.com/docs/api-reference/chat/create for the API specification. 
    """
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    chatbot = router.get_chatbot()

    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=request.max_tokens,
        echo=False,
        stop=request.stop,
    )

    max_new_tokens, error_check_ret = await check_length(
        request,
        gen_params["prompt"],
        gen_params["max_new_tokens"],
        worker_addr,
    )

    if error_check_ret is not None:
        return error_check_ret

    gen_params["max_new_tokens"] = max_new_tokens

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n, worker_addr
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params, worker_addr))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if isinstance(content, str):
            content = json.loads(content)

        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        if "usage" in content:
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, worker_addr: str
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params, worker_addr):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = (
                decoded_unicode
                if len(decoded_unicode) > len(previous_text)
                else previous_text
            )

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    request.prompt = process_input(request.model, request.prompt)

    worker_addr = await get_worker_address(request.model)
    for text in request.prompt:
        max_tokens, error_check_ret = await check_length(
            request, text, request.max_tokens, worker_addr
        )
        if error_check_ret is not None:
            return error_check_ret

        if isinstance(max_tokens, int) and max_tokens < request.max_tokens:
            request.max_tokens = max_tokens

    if request.stream:
        generator = generate_completion_stream_generator(
            request, request.n, worker_addr
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in request.prompt:
            gen_params = await get_gen_params(
                request.model,
                worker_addr,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                max_tokens=request.max_tokens,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
                best_of=request.best_of,
                use_beam_search=request.use_beam_search,
            )
            for i in range(request.n):
                content = asyncio.create_task(
                    generate_completion(gen_params, worker_addr)
                )
                text_completions.append(content)

        try:
            all_tasks = await asyncio.gather(*text_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=create_openai_logprobs(content.get("logprobs", None)),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )


async def generate_completion_stream_generator(
    request: CompletionRequest, n: int, worker_addr: str
):
    model_name = request.model
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(n):
            previous_text = ""
            gen_params = await get_gen_params(
                request.model,
                worker_addr,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                max_tokens=request.max_tokens,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
            )
            async for content in generate_completion_stream(gen_params, worker_addr):
                if content["error_code"] != 0:
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text) :]
                previous_text = (
                    decoded_unicode
                    if len(decoded_unicode) > len(previous_text)
                    else previous_text
                )
                # todo: index is not apparent
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=create_openai_logprobs(content.get("logprobs", None)),
                    finish_reason=content.get("finish_reason", None),
                )
                chunk = CompletionStreamResponse(
                    id=id,
                    object="text_completion",
                    choices=[choice_data],
                    model=model_name,
                )
                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion_stream(payload: Dict[str, Any], worker_addr: str):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            # content = await response.aread()
            buffer = b""
            async for raw_chunk in response.aiter_raw():
                buffer += raw_chunk
                while (chunk_end := buffer.find(delimiter)) >= 0:
                    chunk, buffer = buffer[:chunk_end], buffer[chunk_end + 1 :]
                    if not chunk:
                        continue
                    yield json.loads(chunk.decode())


async def generate_completion(payload: Dict[str, Any], worker_addr: str):
    return await fetch_remote(worker_addr + "/worker_generate", payload, "")


@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
@app.post("/v1/engines/{model_name}/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    request.input = process_input(request.model, request.input)

    data = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i : min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
            "encoding_format": request.encoding_format,
        }
        embedding = await get_embedding(payload)
        if "error_code" in embedding and embedding["error_code"] != 0:
            return create_error_response(embedding["error_code"], embedding["text"])
        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


async def get_embedding(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    worker_addr = await get_worker_address(model_name)

    embedding = await fetch_remote(worker_addr + "/worker_get_embeddings", payload)
    return json.loads(embedding)
