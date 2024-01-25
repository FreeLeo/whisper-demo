from collections import deque
import threading
from tkinter import scrolledtext
from openai import OpenAI
import os
import wave
import pyaudio
from faster_whisper import WhisperModel
from playsound import playsound

import tkinter as tk
import tkinter.font as font

GREEN = "\033[92m"
RESET = "\033[0m"

filter_list = [
    "谢谢大家",
    "由 Amara.org 社群提供的字幕",
    "感谢观看",
    "中文字幕提供",
    "MING PAO CANADA // MING PAO TORONTO",
    "谢谢",
    "中文字幕志愿者",
    "明镜与点点栏目",
    "优优独播剧场——YoYo Television Series Exclusive",
    "響鐘",
    "MING PAO CANADA",
    "谢谢观看"
]

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

chat_log_filename = "log/chatbot_conversation_log.txt"


messages = [
    {"role": "system", "content": "你是一位好朋友，平时和我聊聊天，你一句，我一句，每次不要说太多。"}
]


def get_chat_response(query, model="gpt-3.5-turbo"):
    messages.append({"role": "user", "content": query})
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        chat_response_text = completion.choices[0].message.content
        messages.append({"role": "user", "content": chat_response_text})
        return chat_response_text
    except Exception as e:
        return f"An error occurred: {str(e)}"


def transcribe_chunk(model, chunk_file):
    segments, info = model.transcribe(chunk_file, language="zh",  beam_size=7)
    transcription = ''.join(
        segment.text for segment in segments if segment.text not in filter_list)
    return transcription


def record_chunk(p, stream, file_path, chunk_length=4):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


transcription_buffer = deque(maxlen=100)


def create_ui():
    window = tk.Tk()
    window.title("AI对话")

    customFont = font.Font(family="KaiTi", size=15)

    output_area = scrolledtext.ScrolledText(
        window, wrap=tk.WORD, width=50, height=30, font=customFont)
    output_area.pack(padx=10, pady=10)

    def update_output(text):
        output_area.insert(tk.END, text + '\n')
        output_area.see(tk.END)

    return window, update_output


def transcription_thread(update_output):
    model_size = "large-v3"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = ""

    try:
        while True:
            print("LOG:语音录制中")
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            print("LOG:语音转文字中")
            transcription = transcribe_chunk(model, chunk_file)
            if not transcription:
                continue
            window.after(0, update_output, f"我：{transcription}")
            print(GREEN + transcription + RESET)
            os.remove(chunk_file)

            for char in transcription:
                transcription_buffer.append(char)

            transcription_window = ''.join(transcription_buffer)

            user_input = f"{transcription_window}"
            print("LOG:ChatGPT回答请求中")
            chat_output = get_chat_response(user_input)
            window.after(0, update_output, f"AI：{chat_output}")

            print("LOG:文字转语音中")
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=chat_output,
            )

            response.write_to_file("file/output.mp3")
            print("LOG:AI回答播放中")
            playsound("file/output.mp3")
            os.remove("file/output.mp3")

            accumulated_transcription += transcription + ""
            print("LOG:单次对话结束")

    except KeyboardInterrupt:
        print("Stopping...")
        with open("log/log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG:" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()


def main2():
    global window
    window, update_output = create_ui()

    thread = threading.Thread(
        target=transcription_thread, args=(update_output,))
    thread.daemon = True
    thread.start()

    window.mainloop()


if __name__ == "__main__":
    main2()
