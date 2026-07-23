#scores hybrid against a set
"""
End-to-end romaji eval on Common Voice clips.

Scores each route's audio -> romaji output against a romaji reference built
from the clip's known transcript.

    kanji route :  audio -> Whisper(kanji) -> MeCab      -> romaji
    kana route  :  audio -> Whisper(kana)                -> romaji
    hybrid      :  MeCab, corrected by kana on homographs
    mecab-only  :  hybrid with override disabled (ablation baseline)

CAVEAT worth stating in the writeup: the reference romaji is produced by running
MeCab over the reference TEXT, so MeCab's own reading errors are baked into the
ground truth. This flatters the kanji route slightly. It measures "how close is
the audio pipeline to the text pipeline", not absolute reading correctness --
that's what the hand-labeled reading eval is for.
"""

import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import jiwer

import asr
import route_kanji
import route_kana
import route_hybrid
from homographs import reading_lookup

# ---------------------------------------------------------------- config
BASE = r"C:\Users\dtani\Downloads\1781719747544-cv-corpus-26.0-2026-06-12-ja.tar\cv-corpus-26.0-2026-06-12\ja"
CLIP_DIR = os.path.join(BASE, "clips")
TSV = os.path.join(BASE, "validated.tsv")

N_CLIPS = 50           # start small; bump once it runs clean
OUT_CSV = "route_results.csv"

# override disabled -> pure MeCab, through the identical code path
no_override = lambda word: frozenset()


def reference_romaji(sentence):
    """Ground-truth romaji: MeCab over the reference text."""
    return route_kanji.to_romanji(sentence)


def main():
    df = pd.read_csv(TSV, sep="\t", low_memory=False).head(N_CLIPS)

    rows = []
    for i, r in df.iterrows():
        path = os.path.join(CLIP_DIR, r["path"])
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
    def cer_of(col):
        sub = out[out["reference"].str.strip().astype(bool)
                  & out[col].str.strip().astype(bool)]
        if len(sub) == 0:
            return None, 0
        return jiwer.cer(sub["reference"].tolist(), sub[col].tolist()), len(sub)

    print("\n" + "=" * 46)
    print(f"{'route':<14}{'romaji CER':<14}{'n'}")
    print("-" * 46)
    for col in ["kanji", "kana", "mecab_only", "hybrid"]:
        cer, n = cer_of(col)
        print(f"{col:<14}{cer:<14.3f}{n}" if cer is not None
              else f"{col:<14}{'n/a':<14}{n}")
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