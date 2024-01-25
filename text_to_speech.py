from openai import OpenAI
from playsound import playsound
import time
import os

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello world! This is a streaming test.",
)
response.write_to_file("file/output.mp3")
print("音频1开始播放")
playsound("file/output.mp3")
print("音频1播放完成。")

os.remove("file/output.mp3")

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="锄禾日当午，汗滴禾下土",
)
response.write_to_file("file/output.mp3")
print("音频2开始播放")
playsound("file/output.mp3")
print("音频2播放完成。")
