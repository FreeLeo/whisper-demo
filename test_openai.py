from collections import deque
import threading
from tkinter import scrolledtext
from openai import OpenAI
import os
import wave
import pyaudio
from faster_whisper import WhisperModel

import tkinter as tk
import tkinter.font as font

GREEN = "\033[92m"
RESET = "\033[0m"

OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

chat_log_filename = "log/chatbot_conversation_log.txt"


def get_chat_response(query, model="gpt-3.5-turbo"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Yor are expert sentiment analysist."},
                {"role": "user", "content": query}
            ]
        )
        chat_response_text = completion.choices[0].message.content
        return chat_response_text
    except Exception as e:
        return f"An error occurred: {str(e)}"


def transcribe_chunk(model, chunk_file):
    segments, info = model.transcribe(chunk_file, language="zh",  beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription


def record_chunk(p, stream, file_path, chunk_length=5):
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
    window.title("情感分析")

    customFont = font.Font(family="Helvetica", size=30)

    output_area = scrolledtext.ScrolledText(
        window, wrap=tk.WORD, width=30, height=20, font=customFont)
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
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(GREEN + transcription + RESET)
            os.remove(chunk_file)

            for char in transcription:
                transcription_buffer.append(char)

            transcription_window = ''.join(transcription_buffer)

            user_input = f"对话: {transcription_window}，情感分析一下，回答仅用'积极','中性' or '消极')"
            chat_output = get_chat_response(user_input)
            window.after(0, update_output, f"{chat_output} , {transcription}")

            accumulated_transcription += transcription + ""

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
