# audio samples (numpy) -> Whisper kanji text. Live version of asr.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cuda_setup
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float16")   # large-v3 is the bottleneck

def transcribe_array(audio, initial_prompt=None):
    segments, info = model.transcribe(
        audio, language="ja",
        beam_size=1,                      # default is 5 — greedy is much faster
        condition_on_previous_text=False, # stops drift on repeated partial passes
        vad_filter=False,                 # you already VAD in vad.py; this re-runs Silero every pass
        initial_prompt=initial_prompt,
    )
    return "".join(s.text for s in segments).strip()