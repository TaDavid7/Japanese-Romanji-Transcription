#runs from both routes and runs the format shown in plan (to perfer Mecab, but when disagree reference from another)
"""
for each word:
    if the 2 readings disagree AND the known homograph is listed as a potential pronounciation of the kanji reading
        trust the kana
    else
        keep Mecab

"""
import asr
import fugashi, jaconv
from route_kanji import PARTICLE_SOUND
from route_kana import transcribe_kana
from homographs import reading_lookup
import re
KANJI = re.compile(r"[\u4e00-\u9fff]")


tagger = fugashi.Tagger()
def _skel(s):
    return s.replace("っ", "").replace("ー", "")

def resolve_word(surface, mecab_kana, kana_ref, lookup=reading_lookup):
    # only kanji-bearing tokens can have reading ambiguity worth fixing
    if not KANJI.search(surface):
        return mecab_kana
    if len(surface) < 2:
        return mecab_kana
    valid = lookup(surface)
    if not valid:
        return mecab_kana
    for alt in valid - {mecab_kana}:
        if _skel(alt) in _skel(mecab_kana) or _skel(mecab_kana) in _skel(alt):
            continue
        if len(alt) >= 3 and alt in kana_ref:
            return alt
    return mecab_kana
    

def transcribe(path, lookup=reading_lookup):
    kanji_text = asr.transcribe(path)
    kana_ref = transcribe_kana(path)

    parts = []
    for word in tagger(kanji_text):
        surface = word.surface
        pos = word.feature.pos1
        if pos == "助詞" and surface in PARTICLE_SOUND:
              parts.append(PARTICLE_SOUND[surface])
              continue

        mecab_kana = jaconv.kata2hira(word.feature.kana) if word.feature.kana else surface
        parts.append(resolve_word(surface, mecab_kana, kana_ref, lookup))
    kana_sentence = "".join(parts)
    return jaconv.kana2alphabet(kana_sentence)

