import multiprocessing
import threading
import time
import os
import cv2
import wave
import pyaudio
from faster_whisper import WhisperModel
from playsound import playsound
from pathlib import Path
import google.generativeai as genai
from openai import OpenAI

OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)


def show_camera0(count, stop_camera):
    print("LOG: show_camera_thread start")
    # 打开摄像头（默认摄像头，如果有多个摄像头可以指定编号，例如0、1、2等）
    cap = cv2.VideoCapture(0)
    print("LOG: show_camera_thread start2")
    while True:
        # 读取摄像头画面帧
        ret, frame = cap.read()

        # 显示画面
        cv2.imshow('Camera', frame)
        if count.value >= 10:
            cv2.imwrite('file/screenshot10.jpg', frame)
            count.value = 0

        # 检测按键，如果按下 's' 键则保存当前帧为图片
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 保存当前帧为图片
            cv2.imwrite('file/screenshot.jpg', frame)
            print("截图已保存为 file/screenshot.jpg")

        # 检测按键，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

    stop_camera.value = True
    print(f"LOG: show_camera_thread end. {stop_camera.value}")


def show_camera(count, stop_camera, question, state):
    # 打开摄像头（默认摄像头，如果有多个摄像头可以指定编号，例如0、1、2等）
    cap = cv2.VideoCapture(0)
    while True:
        # 读取摄像头画面帧
        ret, frame = cap.read()

        # 显示画面
        cv2.imshow('Camera', frame)
        if state.value == 1:
            print("截图")
            cv2.imwrite('file/screenshot.jpg', frame)
            state.value = 2

        # 检测按键，如果按下 's' 键则保存当前帧为图片
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 保存当前帧为图片
            cv2.imwrite('file/screenshot.jpg', frame)
            print("截图已保存为 file/screenshot.jpg")

        # 检测按键，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

    stop_camera.value = True
    print(f"LOG: show_camera_thread end. {stop_camera.value}")


# 线程函数，用于增加计数器的值
def increment_count(count, stop_camera):
    while not stop_camera.value:
        time.sleep(1)
        count.value += 1
        print(f"increment_count {stop_camera.value} {count.value}")


filter_list = [
    "谢谢大家",
    "謝謝大家",
    "謝謝",
    "由 Amara.org 社群提供的字幕",
    "感谢观看",
    "中文字幕提供",
    "MING PAO CANADA // MING PAO TORONTO",
    "谢谢",
    "中文字幕志愿者",
    "明镜与点点栏目",
    "优优独播剧场——YoYo Television Series Exclusive",
    "響鐘",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
]


def transcribe_chunk(model, chunk_file):
    text = ""
    segments, info = model.transcribe(
        chunk_file, language="zh",  beam_size=10, vad_filter=False)
    for segment in segments:
        if segment.text not in filter_list:
            text += segment.text
    return text


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


def transcription_question(question, state):
    print("LOG: start...")
    model_size = "large-v3"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=1024)
    try:
        while True:
            if state.value == 2:
                print("state=2, 即将执行ask_gemini")
                answer = ask_gemini(question=question.value)
                play_answer(answer)
                state.value = 0
                continue
            if state == 1:
                time.sleep(1)
                continue
            print("LOG:语音录制中")
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            print("LOG:语音转文字中")
            transcription = transcribe_chunk(model, chunk_file)
            if not transcription.strip() or not transcription.startswith("提问"):
                continue
            print(f"ask {transcription}")
            state.value = 1
            question.value = transcription
            os.remove(chunk_file)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def play_answer(answer):
    print(f"play_answer：{answer}")
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=answer,
    )

    response.write_to_file("file/gemini_output.mp3")
    playsound("file/gemini_output.mp3")
    os.remove("file/gemini_output.mp3")


def ask_gemini(question):
    print(f"询问Gemini: {question}")
    genai.configure(
        api_key="",
        transport="rest")

    # Set up the model
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-pro-vision",
        generation_config=generation_config,
        safety_settings=safety_settings)

    # Validate that an image is present
    if not (img := Path("file/screenshot.jpg")).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path("file/screenshot.jpg").read_bytes()
        },
    ]

    prompt_parts = [
        image_parts[0],
        f"\n{question}",
    ]

    response = model.generate_content(prompt_parts)
    print(f"Gemini回复：{response.text}")
    return response.text


def main2():
    shared_count = multiprocessing.Value('i', 0)
    shared_stop_camera = multiprocessing.Value('b', False)
    manager = multiprocessing.Manager()
    shared_question = manager.Value(str, "")
    """
    0，录音提问中；
    1，录音已转成文字，通知camera截图；
    2.camera截图后，调用gemini，播放tts；
    """
    shared_state = multiprocessing.Value('i', 0)

    show_camera_process = multiprocessing.Process(
        target=show_camera,
        args=(shared_count, shared_stop_camera, shared_question, shared_state))
    show_camera_process.daemon = True
    show_camera_process.start()

    qa_process = multiprocessing.Process(
        target=transcription_question, args=(shared_question, shared_state))
    qa_process.daemon = True
    qa_process.start()
    show_camera_process.join()
    qa_process.join()

    # count_process = multiprocessing.Process(
    #     target=increment_count, args=(shared_count, shared_stop_camera))
    # count_process.daemon = True
    # count_process.start()
    # show_camera_process.join()
    # count_process.join()

    print("LOG: 程序执行结束。")


if __name__ == "__main__":
    main2()
