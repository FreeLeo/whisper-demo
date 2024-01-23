from faster_whisper import WhisperModel
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
print(f"Model load: {time.time()-start_time} s")

start_time = time.time()
segments, info = model.transcribe("file/sss.mp3", beam_size=5)
print(f"Recognition: {time.time()-start_time} s")

print("Detected language '%s' with probability %f" %
      (info.language, info.language_probability))

start_time = time.time()
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
print(f"segments parse: {time.time() - start_time}")
