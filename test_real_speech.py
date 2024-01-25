import os
import wave
import pyaudio
from faster_whisper import WhisperModel


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
    "響鐘"
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


def main2():
    print("LOG: start...")
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
            if not transcription.strip():
                continue
            print(transcription)
            os.remove(chunk_file)

            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")
        with open("log/log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG:" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main2()
