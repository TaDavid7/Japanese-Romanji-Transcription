#shared function. Takes audio file -> Whispher transcription (kanji text.)
import os, sys
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
import cuda_setup
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe(path, initial_prompt=None):
    segments, info = model.transcribe(path, language="ja", vad_filter=True, initial_prompt=initial_prompt)
    return "".join(s.text for s in segments).strip()