#audio -> romanji 
import asr
import jaconv


#using whispher with hirigana prompt

HIRAGANA_PROMPT = "これはぜんぶひらがなでかかれたぶんしょうです"

def transcribe_kana(path):
    kana = asr.transcribe(path, initial_prompt=HIRAGANA_PROMPT)
    return jaconv.kata2hira(kana) #gives hiragana

def transcribe(path):
    kana = asr.transcribe(path, initial_prompt=HIRAGANA_PROMPT) #sends audio to Whisper
    return jaconv.kana2alphabet(jaconv.kata2hira(kana)) #kata2hira changes to hira, and kana2alphabet changes to romanji