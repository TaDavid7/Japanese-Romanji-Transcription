import pandas as pd
df = pd.read_csv("asr_results.csv").dropna()
for _, r in df.iterrows():
    ref, hyp = str(r["reference"]).strip(), str(r["hypothesis"]).strip()
    if ref != hyp:
        print("REF:", ref)
        print("HYP:", hyp)
        print()