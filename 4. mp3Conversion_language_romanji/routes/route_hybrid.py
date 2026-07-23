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

tagger = fugashi.Tagger()

def resolve_word(surface, mecab_kana, kana_ref, lookup=reading_lookup):
    valid = lookup(surface) #get readings of word
    if not valid:
        return mecab_kana
    for alt in valid - {mecab_kana}: #readings that Mecab didn't pick
        if alt in kana_ref:
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