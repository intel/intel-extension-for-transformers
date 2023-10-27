import requests
import os
import httpx
import asyncio
import wave
import time
import base64
 

url = "http://127.0.0.1:8888/v1/talkingbot/llm_tts_chinese"
data = {
    "text": "你是谁",
    # "voice": "pat",
    # "knowledge_id": "default"
}
 
with httpx.stream('POST', url, json=data, verify=False, timeout=200) as r:
    chunk_number = 0
    audio_buffer = b""
    for chunk in r.iter_raw():
        if b"\n\ndata: [DONE]\n\n" in chunk:
            chunk_without_data_done = chunk.split(b"\n\ndata: [DONE]\n\n")[0]
            audio_buffer += chunk_without_data_done
            audio_filename = f"audio_{chunk_number}.wav"
            audio_data = base64.b64decode(audio_buffer)
            with open(audio_filename, "wb") as audio_file:
                audio_file.write(audio_data)
                print("{} generate...".format(audio_filename))
            audio_buffer = b""
        elif b"\n\ndata: b'" in chunk:
            chunk_without_data_end_prefix = chunk.split(b"\n\ndata: b'")[0]
            audio_buffer += chunk_without_data_end_prefix
            audio_filename = f"audio_{chunk_number}.wav"
            audio_data = base64.b64decode(audio_buffer)
            with open(audio_filename, "wb") as audio_file:
                audio_file.write(audio_data)
                print("{} generate...".format(audio_filename))
            chunk_number+=1
            audio_buffer = chunk.split(b"\n\ndata: b'")[1]
        elif b"data: b'" in chunk:
            chunk_without_data_prefix = chunk.split(b"data: b'")[1]
            audio_buffer += chunk_without_data_prefix
        elif b"\n\n" in chunk:
            audio_buffer += chunk.split(b"\n\n")[0]
            audio_filename = f"audio_{chunk_number}.wav"
            audio_data = base64.b64decode(audio_buffer)
            with open(audio_filename, "wb") as audio_file:
                audio_file.write(audio_data)
                print("{} generate...".format(audio_filename))
            audio_buffer = chunk.split(b"\n\n")[1]
            chunk_number+=1
        else:
            audio_buffer += chunk