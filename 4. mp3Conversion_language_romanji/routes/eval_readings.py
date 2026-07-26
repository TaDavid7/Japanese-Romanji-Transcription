#Need to compare
#raw audio -> kanji (whispher) -> romanji (mecab)
#raw audio -> kanji (whispher) -> romanji (pykaksi)
#raw audio -> hiragana (whispher + prompt) -> romanji (jaconv)

import csv

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