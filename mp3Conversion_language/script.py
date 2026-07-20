import cuda_setup
from faster_whisper import WhisperModel
import pandas as pd  #spreadhseet helper
import os   #builds file paths

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

base = r"C:\Users\dtani\Downloads\1781719747544-cv-corpus-26.0-2026-06-12-ja.tar\cv-corpus-26.0-2026-06-12\ja"
clip_directory = base + r"\clips"
tsv = base + r"\validated.tsv"

df = pd.read_csv(tsv, sep="\t") # read the tab sepearated spreadsheet into a table
df = df.head(200)   #keep first 200 rows

rows = []
for i, r in df.iterrows(): # i is the row number, r is the row itself
    path = os.path.join(clip_directory, r["path"]) #gets location of .mp3
    segments, _ = model.transcribe(path, language="ja", vad_filter = True) #run whisper, segment gets transcribed
    #vad filter to prevent ご視聴ありがとうございました when theres nothing
    hypothesis = "".join(s.text for s in segments).strip() #combines into string, trim spaces
    rows.append({"reference": r["sentence"], "hypothesis": hypothesis}) #save pair
    print(i, hypothesis) 

pd.DataFrame(rows).to_csv("asr_results.csv", index=False, encoding="utf-8-sig")
print("done")
