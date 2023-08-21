import argparse
from collections import defaultdict
import datetime
import json
import os
import time
import uuid

os.system("pip install gradio==3.28.0")

import gradio as gr
import requests

from fastchat.conversation import (
    Conversation,
    compute_skip_echo_len,
    SeparatorStyle,
)
from fastchat.constants import LOGDIR
from fastchat.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_css import code_highlight_css


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "NeuralChat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

controller_url = None
enable_moderation = False

conv_template_bf16 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
    sep2="<|endoftext|>",
)

conv_template_bf16 = Conversation(
    system="",
    roles=("### Human", "### Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
    sep2="</s>",
)
# conv_template_bf16 = Conversation(
#     system="",
#     roles=("", ""),
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.OASST_PYTHIA,
#     sep=" ",
#     sep2="<|endoftext|>",
# )

start_message = """<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>"""

conv_template_bf16 = Conversation(
    system=start_message,
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="\n",
    sep2="<|im_end|>",
)

def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""


def load_demo_single(models, url_params):
    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 5


def add_text(state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")

    if state is None:
        state = conv_template_bf16.copy()

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5
    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg) + (
                no_change_btn,
            ) * 5

    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(state, model_selector, temperature, max_new_tokens, topk, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    temperature = float(temperature)
    max_new_tokens = int(max_new_tokens)
    topk = int(topk)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        new_state = conv_template_bf16.copy()
        new_state.conv_id = uuid.uuid4().hex
        new_state.model_name = state.model_name or model_selector
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()
    skip_echo_len = compute_skip_echo_len(model_name, state, prompt)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "topk": topk,
        "stop": "<|endoftext|>"
    }
    logger.info(f"==== request ====\n{pload}")

    start_time = time.time()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            controller_url + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=20,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][skip_echo_len:].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.005)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time() - start_time
    elapsed_time = "\n✅generation elapsed time: {}s".format(round(finish_tstamp, 4))

    # elapsed_time =  "\n{}s".format(round(finish_tstamp, 4))
    # elapsed_time =  "<p class='time-style'>{}s </p>".format(round(finish_tstamp, 4))

    state.messages[-1][-1] = state.messages[-1][-1][:-1] + elapsed_time
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "topk": topk,
            },
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


block_css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}

#notice_markdown {
    text-align: center;
    background: #2e78c4;
    padding: 1%;
    height: 4.3rem;
    color: #fff !important;
    margin-top: 0;
}

#notice_markdown p{
    color: #fff !important;
}


#notice_markdown h1, #notice_markdown h4 {
    color: #fff;
    margin-top: 0;
}

gradio-app {
    background: linear-gradient(to bottom, #86ccf5, #3273bf) !important;
    padding: 3%;
}

.gradio-container {
    margin: 0 auto !important;
    width: 70% !important;
    padding: 0 !important;
    background: #fff !important;
    border-radius: 5px !important;
}

#chatbot {
    border-style: solid;
    overflow: visible;
    margin: 1% 4%;
    width: 90%;
    box-shadow: 0 15px 15px -5px rgba(0, 0, 0, 0.2);
    border: 1px solid #ddd;
}

#chatbot::before {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 60px;
    height: 60px;
    background-image: url(https://i.postimg.cc/gJzQTQPd/Microsoft-Teams-image-73.png);
    background-repeat: no-repeat;
    background-position: center center;
    background-size: contain;
}

#chatbot .wrap {
    margin-top: 30px !important;
}


#text-box-style, #btn-style {
    width: 90%;
    margin: 1% 4%;
}


.user, .bot {
    width: 80% !important;
    
}

.bot {
    white-space: pre-wrap !important;  
    line-height: 1.3 !important;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;

}

#btn-send-style {
    background: rgb(0, 180, 50);
    color: #fff;
    }

#btn-list-style {
    background: #eee0;
    border: 1px solid #0053f4;
}        

.title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff !important;
    display: flex;
    justify-content: center;
}

footer {
    display: none !important;
}

.footer {
    margin-top: 2rem !important;
    text-align: center;
    border-bottom: 1px solid #e5e5e5;
}

.footer>p {
    font-size: .8rem;
    display: inline-block;
    padding: 0 10px;
    transform: translateY(10px);
    background: white;
}

.img-logo {
    width: 3.3rem;
    display: inline-block;
    margin-right: 1rem;
}

.img-logo-style {
    width: 3.5rem;
    float: left;
}

.img-logo-right-style {
    width: 3.5rem;
    float: right;
    margin-top: -1rem;
    margin-left: 1rem;
}

.neural-studio-img-style {
     width: 50%;
    height: 20%;
    margin: 0 auto;
}

.acknowledgments {
    margin-bottom: 1rem !important;
    height: 1rem;
}
"""
)


def build_single_model_ui(models):
 
    notice_markdown = """
<div class="title">
<div style="
    color: #fff;
">Large Language Model <p style="
    font-size: 0.8rem;
">4th Gen Intel® Xeon® with Intel® AMX</p></div>
 
</div>
"""

    learn_more_markdown = """<div class="footer">
                    <p>Powered by <a href="https://github.com/intel/intel-extension-for-transformers" style="text-decoration: underline;" target="_blank">Intel Extension for Transformers</a> and <a href="https://github.com/intel/intel-extension-for-pytorch" style="text-decoration: underline;" target="_blank">Intel Extension for PyTorch</a>
                    <img src='https://i.postimg.cc/Pfv4vV6R/Microsoft-Teams-image-23.png' class='img-logo-right-style'/></p>
            </div>
            <div class="acknowledgments">
            <p></p></div>
            
        """

    state = gr.State()
    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row", visible=False):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
        ).style(container=False)

    chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=550)
    with gr.Row(elem_id="text-box-style"):
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False, elem_id="btn-send-style")

    with gr.Accordion("Parameters", open=False, visible=False, elem_id="btn-style") as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.001,
            step=0.1,
            interactive=True,
            label="Temperature",
            visible=False,
        )
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=1024,
            value=512,
            step=1,
            interactive=True,
            label="Max output tokens",
        )
        topk = gr.Slider(
            minimum=1,
            maximum=10,
            value=1,
            step=1,
            interactive=True,
            label="TOP K",
        )


    with gr.Row(visible=False, elem_id="btn-style") as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False, visible=False, elem_id="btn-list-style")
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False, visible=False, elem_id="btn-list-style")
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False, visible=False, elem_id="btn-list-style")
        # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False, elem_id="btn-list-style")
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False, elem_id="btn-list-style")


    gr.Markdown(learn_more_markdown)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens, topk],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens, topk],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens, topk],
        [state, chatbot] + btn_list,
    )

    return state, model_selector, chatbot, textbox, send_btn, button_row, parameter_row


def build_demo(models):
    with gr.Blocks(
        title="NeuralChat · Intel",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        (
            state,
            model_selector,
            chatbot,
            textbox,
            send_btn,
            button_row,
            parameter_row,
        ) = build_single_model_ui(models)

        if model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [
                    state,
                    model_selector,
                    chatbot,
                    textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ],
                _js=get_window_url_params,
            )
        else:
            raise ValueError(f"Unknown model list mode: {model_list_mode}")

    return demo


if __name__ == "__main__":

    controller_url = "http://54.166.144.154:80"
    host = "0.0.0.0"

    concurrency_count = 10
    model_list_mode = "once"
    share = False
    moderate = False

    set_global_vars(controller_url, moderate)
    models = get_model_list(controller_url)

    demo = build_demo(models)
    demo.queue(
        concurrency_count=concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=host, share=share, max_threads=200
    )
