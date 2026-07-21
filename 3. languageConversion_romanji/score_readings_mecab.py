import fugashi, pandas as pd, jaconv

tagger = fugashi.Tagger()

def reading_in_context(sentence, target):
    idx = sentence.find(target)
    if idx == -1:
        return None
    pos, parts = 0, []
    for word in tagger(sentence):
        w = word.surface
        start, end = pos, pos + len(w)
        if start < idx + len(target) and end > idx:   # token overlaps target span
            k = word.feature.kana
            if k:
                parts.append(jaconv.kata2hira(k))
        pos = end
    return "".join(parts) if parts else None

df = pd.read_csv("readings.csv")
correct = 0
for _, r in df.iterrows():
    got = reading_in_context(r["sentence"], r["target"])
    want = r["correct_reading"]
    hit = (got == want)
    correct += hit
    if not hit:
        print(f"MISS  {r['target']:<4} got:{got}  want:{want}  ({r['sentence']})")

print(f"\nMeCab reading accuracy: {correct/len(df):.3f}  ({correct}/{len(df)})")