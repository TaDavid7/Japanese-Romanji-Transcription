# audio samples (numpy) -> Whisper kanji text. Live version of asr.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cuda_setup
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe_array(audio, initial_prompt=None):   # instead of taking .mp3 file on disk, take audio as raw samples from memory
      # audio = float32 numpy, mono, 16000 Hz, values in [-1, 1]
      segments, info = model.transcribe(
          audio, language="ja", vad_filter=True, initial_prompt=initial_prompt
      )
      return "".join(s.text for s in segments).strip()