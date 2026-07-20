import pykakasi, jiwer, pandas as pd

kks = pykakasi.kakasi()
def to_kana(s):
    return "".join(x["hira"] for x in kks.convert(str(s)))

df = pd.read_csv("asr_results.csv").dropna()
df = df[df["reference"].str.strip().astype(bool)]
df = df[df["hypothesis"].str.strip().astype(bool)]

refs = [to_kana(x) for x in df["reference"]]
hyps = [to_kana(x) for x in df["hypothesis"]]

print(f"scoring {len(refs)} clips")
print(f"reading-level CER: {jiwer.cer(refs, hyps):.3f}")