import jiwer, pandas as pd, re

df = pd.read_csv("asr_results.csv")

# drop rows where either column is empty/blank
df = df.dropna(subset=["reference", "hypothesis"])
df = df[df["reference"].str.strip().astype(bool)]
df = df[df["hypothesis"].str.strip().astype(bool)]

# normalization: strip punctuation and spaces from both sides
def norm(s):
    return re.sub(r"[。、！？「」『』・　\s,.!?\"'’]", "", str(s)).strip()

refs = [norm(x) for x in df["reference"]]
hyps = [norm(x) for x in df["hypothesis"]]

print(f"scoring {len(df)} clips")
# raw vs normalized, so you can report both
raw_cer = jiwer.cer(df["reference"].tolist(), df["hypothesis"].tolist())
norm_cer = jiwer.cer(refs, hyps)
print(f"raw CER:        {raw_cer:.3f}")
print(f"normalized CER: {norm_cer:.3f}")

