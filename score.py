import jiwer, pandas as pd

df = pd.read_csv("asr_results.csv")
cer = jiwer.cer(df["reference"].tolist(), df["hypothesis"].tolist())
print(f"CER: {cer:.3f}")