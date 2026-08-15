# kanji text -> romanji, same MeCab logic as folder 4's route_kanji.py
import fugashi, jaconv

PARTICLE_SOUND = {"は": "わ", "へ": "え", "を": "お"}
tagger = fugashi.Tagger()

def to_romanji(sentence):
    parts = []
    for word in tagger(sentence):
        surface = word.surface
        if word.feature.pos1 == "助詞" and surface in PARTICLE_SOUND:
            parts.append(PARTICLE_SOUND[surface]); continue
        k = word.feature.kana
        parts.append(jaconv.kata2hira(k) if k else surface)
    return jaconv.kana2alphabet("".join(parts))