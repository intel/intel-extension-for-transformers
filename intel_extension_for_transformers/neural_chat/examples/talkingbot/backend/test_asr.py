import requests
import os
import httpx
import asyncio
import wave
import time
import base64
 
#url = "https://52.7.120.207/talkingbot/asr"
 
url = "http://localhost:8888/v1/talkingbot/asr_chinese"
# url = "http://10.165.59.104:8888/v1/talkingbot/asr_chinese"
file_path = "./whoareu_cn.mp3"  # Replace with the actual audio file path
 
 
# Open the audio file in binary mode and read its contents
with open(file_path, "rb") as wav_file:
    files = {
        "file": ("audio.wav", wav_file, "audio/wav"),
    }
    response = requests.post(url, files=files, verify=False)
 
if response.status_code != 200:
    raise Exception("response audio generated failed with status code:" + str(response.status_code))
 
 
print("File uploaded successfully")
print("asr text=", response.text)