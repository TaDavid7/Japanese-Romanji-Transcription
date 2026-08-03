#Need to compare
#raw audio -> kanji (whispher) -> romanji (mecab)
#raw audio -> kanji (whispher) -> romanji (pykaksi)
#raw audio -> hiragana (whispher + prompt) -> romanji (jaconv)
#raw audio -> hybrid -> romanji

"""
End-to-end romaji eval on Common Voice clips.

Scores each route's audio -> romaji output against a romaji reference built
from the clip's known transcript (MeCab over the reference text), so
MeCab's own reading errors are baked into the ground truth. This measures
"how close is the audio pipeline to the text pipeline", not absolute
reading correctness.
"""

import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import asr
import route_kanji
import route_kana
import route_hybrid
from homographs import reading_lookup

# ---------------------------------------------------------------- config
base = r"C:\Users\dtani\Downloads\1781719747544-cv-corpus-26.0-2026-06-12-ja.tar\cv-corpus-26.0-2026-06-12\ja"
clip_directory = base + r"\clips"
tsv = base + r"\validated.tsv"

N_CLIPS = 50           # start small; bump once it runs clean
OUT_CSV = "route_results.csv"

# override disabled -> pure MeCab, through the identical code path
no_override = lambda word: frozenset()


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


def reference_romaji(sentence):
    """Ground-truth romaji: MeCab over the reference text."""
    return route_kanji.to_romanji(sentence)


def main():
    df = pd.read_csv(tsv, sep="\t", low_memory=False).head(N_CLIPS)

    rows = []
    for i, r in df.iterrows():
        path = os.path.join(clip_directory, r["path"])
        if not os.path.exists(path):
            continue

        ref_text = str(r["sentence"])
        ref = reference_romaji(ref_text)

        t0 = time.time()
        kanji_text = route_kanji.to_romanji(asr.transcribe(path))
        kana_out = route_kana.transcribe(path)
        hybrid_out = route_hybrid.transcribe(path, lookup=reading_lookup)
        mecab_only = route_hybrid.transcribe(path, lookup=no_override)
        elapsed = time.time() - t0

        rows.append({
            "clip": r["path"],
            "ref_text": ref_text,
            "reference": ref,
            "kanji": kanji_text,
            "kana": kana_out,
            "hybrid": hybrid_out,
            "mecab_only": mecab_only,
        })
        print(f"[{i}] {elapsed:.1f}s  {ref_text[:30]}")

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
    for col in ["kanji", "kana", "mecab_only", "hybrid"]:
        avg, n = score(col)
        print(f"{col:<14}{avg:<16.3f}{n}" if avg is not None
              else f"{col:<14}{'n/a':<16}{n}")
    print("=" * 46)

    # how often did the override actually fire?
    changed = (out["hybrid"] != out["mecab_only"]).sum()
    print(f"\nhybrid differed from mecab-only on {changed}/{len(out)} clips")
    if changed:
        print("\nexamples where the override fired:")
        diff = out[out["hybrid"] != out["mecab_only"]].head(5)
        for _, d in diff.iterrows():
            print(f"  text  : {d['ref_text'][:40]}")
            print(f"  mecab : {d['mecab_only'][:60]}")
            print(f"  hybrid: {d['hybrid'][:60]}\n")


if __name__ == "__main__":
    main()
