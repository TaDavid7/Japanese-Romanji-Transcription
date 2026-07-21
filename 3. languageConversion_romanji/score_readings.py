import pykakasi, pandas as pd, jaconv

kks = pykakasi.kakasi()

def reading_in_context(sentence, target):
    # pykakasi is token-level and ignores context — but we still read the
    # target's span so the comparison is structured the same as MeCab's
    idx = sentence.find(target)
    if idx == -1:
        return None
    parts = kks.convert(target)
    return "".join(jaconv.kata2hira(p["kana"]) for p in parts)

df = pd.read_csv("readings.csv")
correct = 0
for _, r in df.iterrows():
    got = reading_in_context(r["sentence"], r["target"])
    want = r["correct_reading"]
    hit = (got == want)
    correct += hit
    if not hit:
        print(f"MISS  {r['target']:<4} got:{got}  want:{want}  ({r['sentence']})")

print(f"\npykakasi reading accuracy: {correct/len(df):.3f}  ({correct}/{len(df)})")