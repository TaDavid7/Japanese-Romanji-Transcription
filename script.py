import cuda_setup
from faster_whisper import WhisperModel
import pykakasi

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
kks = pykakasi.kakasi()

def transcribe(path):
    segments, _ = model.transcribe(path, language="ja")
    text = "".join(s.text for s in segments)
    romaji = " ".join(item["hepburn"] for item in kks.convert(text))
    return text, romaji

if __name__ == "__main__":
    print(transcribe("test.mp3"))