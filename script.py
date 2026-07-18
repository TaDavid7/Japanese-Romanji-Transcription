import cuda_setup
from faster_whisper import WhisperModel
import pykakasi
import pandas as pd
import os

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

clip_directory = "clips"
tsv = "validated.tsv"

df = pd.read_csv(tsv, sep="\t")
df = df.head(200)

rows = []
for i, r in df.iterrows():
    path = os.path.join(clip_directory, r["path"])
    segments, _ = model.transcribe(path, language="ja")
    hypothesis = "".join(s.text for s in segments).strip()
    rows.append({"reference": r["sentence"], "hypothesis": hypothesis})
    print(i, hypothesis)

pd.DataFrame(rows).to_csv("asr_results.csv", index=False, encoding="utf-8-sig")
print("done")
#Setup for using pykakasi to translate one mp3 file
"""
kks = pykakasi.kakasi()
def transcribe(path):
    segments, _ = model.transcribe(path, language="ja")
    text = "".join(s.text for s in segments)
    romaji = " ".join(item["hepburn"] for item in kks.convert(text))
    return text, romaji

if __name__ == "__main__":
    print(transcribe("test.mp3"))
"""