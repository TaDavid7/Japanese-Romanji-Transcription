from functools import lru_cache #a memory so it remembers what a function returned
import jaconv

_jam = None #placeholder for dictionary connection, nothing loading, protected don't touch from outside

#Lazy loading, don't import until needed
def _get_jam():
    global _jam
    if _jam is None:
        from jamdict import Jamdict
        _jam = Jamdict()
    return _jam

#Lookup
@lru_cache(maxsize=4096) # look up 人気 once, the answer's cached, ask again it's instant. maxsize=4096 = remember up to 4096 words
#readings_of("人気")   not seen before → do the work, store the answer
#readings_of("人気")   seen it → return stored answer instantly
def readings_of(word, common_only=False):
    try:
        result = _get_jam().lookup(word)
    except Exception:
        return frozenset()
    out = set()
    for entry in result.entries:
        for kana_form in entry.kana_forms:
            out.add(jaconv.kata2hira(kana_form.text)) #gets hiragana readings of that word
    return frozenset(out) #return set of words

#ambiguity test
def is_ambiguous(word, min_readings=2):
    return len(readings_of(word)) >= min_readings

#interface
def reading_lookup(word):
    r = readings_of(word)
    return r if len(r) >=2 else frozenset()

