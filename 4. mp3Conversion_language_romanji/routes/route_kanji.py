#kanji text -> romanji via Mecab
#same logic as 3. revision
import asr
import fugashi, jaconv #fugashi runs MeCab (kanji -> katakana), and jaconv (katakana -> hiragana)

PARTICLE_SOUND = {"は": "わ", "へ": "え", "を": "お"}

tagger = fugashi.Tagger() # creates a MeCab tagger instance
def to_kana(sentence):
    parts = []
    for word in tagger(sentence): #tagger(sentence) is run Mecab's tokernizer on this string and is separable by tokens
        k = word.feature.kana #each word carries .feature is dict info and .kana is the dictionary's recorded pronounciation
        parts.append(jaconv.kata2hira(k) if k else word.surface) #if k has smth, convert it, otherwise give original token
    return "".join(parts)

def to_kana_spoken(sentence):
    parts = []
    for word in tagger(sentence):
        surface = word.surface
        pos = word.feature.pos1 
        if pos == "助詞" and surface in PARTICLE_SOUND:   # 助詞 = particle
            parts.append(PARTICLE_SOUND[surface])
            continue
        k = word.feature.kana
        parts.append(jaconv.kata2hira(k) if k else surface)
    return "".join(parts)

def to_romanji(sentence):
    return jaconv.kana2alphabet(to_kana_spoken(sentence))