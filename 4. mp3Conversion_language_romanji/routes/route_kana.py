#audio -> romanji 
import asr
import jaconv


#using whispher with hirigana prompt

HIRAGANA_PROMPT = "これはぜんぶひらがなでかかれたぶんしょうです"

def transcribe(path):
    kana = asr.transcribe(path, initial_prompt=HIRAGANA_PROMPT)
    return jaconv.kana2alphabet(jaconv.kata2hira(kana))