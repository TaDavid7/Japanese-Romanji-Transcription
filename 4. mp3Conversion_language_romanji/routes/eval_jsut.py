#audio is under jsut_ver1.1/jsut_ver1.1/basic5000/wav
#transcription is under jsut-label/text_kana/basic5000.yml (using kana_level3 for each one)
#right now the .yml only has kana and not romanji, but since its kana, change to romanji and compare that

#run hybrid, kana, kanji routes on 200 of the first wav clips and compare accuracy based on confirm transcription in .yml
#make sure to strip punctuation or spaces or any non obvious characters to make sure it matches

"""
End-to-end romaji eval on the JSUT basic5000 corpus.

Scores each route's audio -> romaji output against a romaji reference built
from kana_level3 in jsut-label (jaconv over the known-correct kana), so this
measures how close each audio pipeline gets to a verified reading, rather
than "how close is audio to MeCab's own guess" (eval_readings.py).
"""

import os, re, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jaconv
import pandas as pd
import yaml

import asr
import route_kanji
import route_kana
import route_hybrid
from homographs import reading_lookup

# ---------------------------------------------------------------- config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
wav_dir = os.path.join(project_root, "jsut_ver1.1", "jsut_ver1.1", "basic5000", "wav")
yaml_path = os.path.join(project_root, "jsut-label", "text_kana", "basic5000.yaml")

N_CLIPS = 500
OUT_CSV = "jsut_results.csv"

LONG = [("a-", "aa"), ("i-", "ii"), ("u-", "uu"), ("e-", "ee"), ("o-", "oo")]

def canon(s):
    s = s.lower()
    for a, b in LONG:                      # jaconv's chouon hyphen
        s = s.replace(a, b)
    s = s.replace("ou", "oo").replace("ei", "ee")   # spelled-out long vowels
    s = s.replace("wo", "o")               # particle を convention
    return re.sub(r"[^a-z]", "", s)

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j-1] + 1,    # insertion
                prev[j-1] + cost  # substitution
            )
        prev = curr
    return prev[-1]


def similarity(ref, hyp):
    """1.0 = identical, 0.0 = completely different. None if ref is empty."""
    if not ref:
        return None
    return 1 - levenshtein(ref, hyp) / max(len(ref), len(hyp))

def main():
    with open(yaml_path, encoding="utf-8") as f:
        labels = yaml.safe_load(f)

    keys = list(labels.keys())[:N_CLIPS]

    rows = []
    for i, key in enumerate(keys):
        path = os.path.join(wav_dir, f"{key}.wav")
        if not os.path.exists(path):
            continue

        ref = canon(jaconv.kana2alphabet(labels[key]["kana_level3"]))

        t0 = time.time()
        kanji_out  = canon(route_kanji.to_romanji(asr.transcribe(path)))
        kana_out   = canon(route_kana.transcribe(path))
        hybrid_out = canon(route_hybrid.transcribe(path, lookup=reading_lookup))
        elapsed = time.time() - t0

        rows.append({
            "clip": key,
            "reference": ref,
            "kanji": kanji_out,
            "kana": kana_out,
            "hybrid": hybrid_out,
        })
        print(f"[{i}] {elapsed:.1f}s  {key}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- scoring
    def score(col):
        sims = [similarity(ref, hyp) for ref, hyp in zip(out["reference"], out[col])]
        sims = [s for s in sims if s is not None]
        if not sims:
            return None, 0
        return sum(sims) / len(sims), len(sims)

    print("\n" + "=" * 46)
    print(f"{'route':<14}{'avg similarity':<16}{'n'}")
    print("-" * 46)
    for col in ["kanji", "kana", "hybrid"]:
        avg, n = score(col)
        print(f"{col:<14}{avg:<16.3f}{n}" if avg is not None
              else f"{col:<14}{'n/a':<16}{n}")
    print("=" * 46)

    n_same = (out["kanji"] == out["hybrid"]).sum()
    print(f"\nkanji == hybrid on {n_same}/{len(out)} clips")
    if n_same == len(out):
        print("WARNING: hybrid produced no distinct output — lookup never fired")

    try:
        from route_hybrid import STATS
        print("hybrid STATS:", STATS)
    except ImportError:
        pass


if __name__ == "__main__":
    main()