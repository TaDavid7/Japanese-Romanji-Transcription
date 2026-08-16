# Japanese Speech → Rōmaji Transcription

Turns Japanese audio into **rōmaji** (Latin-script pronunciation), from both audio files and
a live microphone stream.

The hard part isn't the speech recognition — it's the last step. Japanese text doesn't tell you
how it's pronounced. This project builds and benchmarks three different pipelines for that step,
then ships the winner as a real-time transcriber.

```
市場で野菜を買う   →   ichiba de yasai o kau      ("marketplace")
市場が動く         →   shijou ga ugoku            ("financial market")
```

Same kanji, two readings. Picking the right one is the whole problem.

---

## Contents

- [Why this is hard](#why-this-is-hard)
- [Architecture](#architecture)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Repo layout](#repo-layout)
- [Design notes](#design-notes)
- [Limitations & next steps](#limitations--next-steps)

---

## Why this is hard

A naive pipeline is `audio → Whisper → text → romanize`. Three things break it:

**1. Whisper outputs kanji, and kanji is not phonetic.**
`人気` is `にんき` (*ninki*, "popularity") or `ひとけ` (*hitoke*, "signs of life") depending
entirely on context. A character-level romanizer has no way to choose.

**2. You can't just ask Whisper for kana instead.**
Prompting Whisper to emit hiragana works, but it throws away the kanji that disambiguates
*meaning* — and empirically it's a much less accurate transcription (0.878 vs 0.978 similarity,
[see below](#route-comparison-jsut-basic5000-500-clips)).

**3. Scoring is a trap.**
Comparing `私` against `わたし` as raw text scores as a total miss even though the pronunciation
is identical. Character Error Rate on kanji text measures the wrong thing — you have to
normalize both sides to a phonetic representation before scoring, or your metric punishes
correct answers.

## Architecture

Two entry points share the same romanization core.

### Batch (audio file → rōmaji)

```
                          ┌──────────────────────────────────────┐
   audio file ──────────► │ faster-whisper large-v3  (+ Silero VAD) │
                          └──────────────────────────────────────┘
                                │                        │
                    kanji text  │                        │  hiragana text
                                │                        │  (via hiragana prompt)
                                ▼                        ▼
                     ┌───────────────────┐     ┌───────────────────┐
                     │ MeCab / UniDic    │     │  direct kana      │
                     │ tokenize + read   │     │  → rōmaji         │
                     └───────────────────┘     └───────────────────┘
                                │                        │
                                └────────┬───────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │  hybrid resolver    │  JMdict homograph lookup:
                              │  (route_hybrid.py)  │  when the two disagree AND
                              └─────────────────────┘  the alternative is a known
                                         │             reading, trust the kana
                                         ▼
                                      rōmaji
```

### Live (microphone → rōmaji, streaming)

```
  mic ──► ring buffer ──► energy VAD ──► Whisper medium ──► LocalAgreement-2 ──► MeCab ──► rōmaji
       (0.1s blocks,      (RMS         (re-runs on the      (commit only text     (incremental,
        queue-backed)      threshold)   growing buffer       that survived two     in-place
                                        every 1.0s)          consecutive runs)     terminal render)
```

The interesting piece is **LocalAgreement-2**. Re-transcribing a growing buffer makes Whisper
revise its own earlier output, so naive streaming flickers badly. LocalAgreement only commits
the longest common prefix between two consecutive hypotheses — text that two independent passes
agreed on. Committed text never changes; only the tail is volatile.

## Results

### Stage 2 — driving down transcription error

Common Voice Japanese, 200 clips, Character Error Rate (lower is better):

| Change | CER |
|---|---|
| Baseline, raw comparison | 0.425 |
| + strip punctuation / normalize formatting | 0.399 |
| + compare on kana rather than kanji (`私` ≡ `わたし`) | 0.362 |
| + Silero VAD filter (kills hallucinated text on silence) | **0.203** |

Over half the apparent "error" was measurement artifact, not model error. The VAD fix alone
cut CER by 44% — without VAD, Whisper hallucinates stock phrases like
`ご視聴ありがとうございました` ("thanks for watching", learned from YouTube subtitles) over
silent segments.

### Stage 3 — reading accuracy on hand-written hard cases

20 sentences written specifically to contain ambiguous readings:

| Romanizer | Accuracy |
|---|---|
| pykakasi (character-level) | 0.650 (13/20) |
| MeCab + UniDic (morphological) | **0.750 (15/20)** |

MeCab's remaining 5 failures, categorized:

| Target | Got | Wanted | Cause |
|---|---|---|---|
| 一日 | ついたち | いちにち | insufficient context |
| 人気 | にんき | ひとけ | genuinely undeterminable from text alone |
| 市場 | しじょう | いちば | insufficient context |
| 方 | ほう | かた | grammatical (honorific usage) |
| 一冊 | いちさつ | いっさつ | counter-word euphony (sokuon) |

The `人気` case is what motivated the hybrid route: the *audio* contains the answer even
though the *text* doesn't.

### Route comparison (JSUT basic5000, 500 clips)

Scored as normalized Levenshtein similarity against `kana_level3` ground-truth annotations
from [jsut-label](https://github.com/sarulab-speech/jsut-label) (higher is better):

| Route | Avg similarity | n |
|---|---|---|
| **kanji** (Whisper → MeCab) | **0.978** | 500 |
| kana (Whisper w/ hiragana prompt) | 0.878 | 500 |
| **hybrid** (kanji, kana-arbitrated) | **0.978** | 500 |

`kanji == hybrid` on 487/500 clips.

**Honest conclusion: the hybrid route doesn't pay for itself.** It fires rarely, and when it
does it's roughly break-even — 7 wins against 6 losses on this run:

| | Clip | Change |
|---|---|---|
| ✅ | 0011 | 日本 `nippon` → `nihon` |
| ✅ | 0077 | 闇夜 `anya` → `yamiyo` |
| ✅ | 0364 | 瞬く `shibatataku` → `matataku` |
| ✅ | 0342 | 早急 `sakkyuu` → `sookyuu` |
| ❌ | 0010 | `hontoo` → `honto` |
| ❌ | 0335 | `akanboo` → `akanbo` |
| ❌ | 0186 | `kesshite` → `keshite` |

Every loss is the same bug class: JMdict lists a shortened orthographic variant as a legitimate
reading, and the resolver takes it. The `_skel()` guard in `route_hybrid.py` (which strips っ and
ー before comparing, so variants of the same word don't count as disagreement) was added for
exactly this and catches most of it. The remaining losses need vowel-length normalization too.

Getting an earlier version of this from *actively harmful* (0.929) to *break-even* (0.978)
required three fixes: a minimum-length guard on substitutions, sokuon handling (MeCab emits っ
as its own token, which JMdict would "resolve" into the literal string `xtsu`), and consistent
romanization conventions on both sides of the comparison.

**The live pipeline therefore uses the plain kanji route.** Same accuracy, half the inference cost.

## Setup

Requires **Python 3.12** and an **NVIDIA GPU** (CUDA). Whisper `large-v3` in float16 wants
~5GB VRAM; the live pipeline uses `medium` (~2.5GB).

### 1. Environment

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS the activate line is `source venv/bin/activate`.

### 2. CUDA libraries

`faster-whisper` needs cuBLAS and cuDNN on the DLL path. `requirements.txt` installs them as
pip packages, and `cuda_setup.py` (imported first by every entry point) adds them to `PATH` at
runtime, so no system-wide CUDA install is needed:

```python
# cuda_setup.py — imported before faster_whisper, must stay first
base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
for pkg in ["cublas/bin", "cudnn/bin"]:
    os.add_dll_directory(os.path.join(base, *pkg.split("/")))
```

> This path is Windows-specific. On Linux, set `LD_LIBRARY_PATH` to the equivalent
> `site-packages/nvidia/*/lib` directories instead.

To run on CPU instead, change `device="cuda", compute_type="float16"` to
`device="cpu", compute_type="int8"` in `asr.py` / `asr_live.py`. It works, but it's slow enough
that the live pipeline won't keep up with real time.

### 3. JMdict database (only needed for the hybrid route)

`pip install jamdict-data` frequently fails to decompress its bundled database. Install it manually:

```powershell
mkdir C:\temp\jd -Force; cd C:\temp\jd

# download the sdist without installing it
python -c "import urllib.request as u, json; d=json.load(u.urlopen('https://pypi.org/pypi/jamdict-data/1.5/json')); url=[f['url'] for f in d['urls'] if f['filename'].endswith('.tar.gz')][0]; u.urlretrieve(url, 'jamdict_data-1.5.tar.gz')"

tar -xf jamdict_data-1.5.tar.gz
cd jamdict_data-1.5\jamdict_data

# decompress the db — this is the step the installer fumbles
python -c "import lzma, shutil; shutil.copyfileobj(lzma.open('jamdict.db.xz'), open('jamdict.db','wb'))"

# put it where jamdict looks by default
mkdir $env:USERPROFILE\.jamdict\data -Force
copy jamdict.db $env:USERPROFILE\.jamdict\data\jamdict.db
```

`homographs.py` calls `lookup("行く")` on first use specifically to fail loudly if this is missing —
a silent failure here makes the hybrid route degrade invisibly into the plain kanji route.

### 4. Evaluation datasets (only needed to reproduce the benchmarks)

```bash
# JSUT audio — basic5000 subset
wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
unzip jsut_ver1.1.zip

# kana annotations (the zip ships kanji only, so this clone is required)
git clone https://github.com/sarulab-speech/jsut-label.git
```

Both land in the repo root and are gitignored. Common Voice Japanese (used by stage 2) needs a
manual download from [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets); the
extracted path is hardcoded at the top of `2. mp3Conversion_language/script.py`.

## Usage

### Live microphone transcription

```powershell
cd "5.mp3Conversion_romanji_live"
python live.py
```

Speak into the default input device. Confirmed text stays put; the current utterance updates in
place and is flushed to its own line after 0.8s of silence.

```
Listening... (Ctrl+C to stop)

kyou wa ii tenki desu ne
ashita no yotei wa mada kimattemasen
```

Tunables at the top of `live.py`:

| Constant | Default | Effect |
|---|---|---|
| `STEP_SECONDS` | 1.0 | How often Whisper re-runs. Lower = more responsive, more CPU, more flicker. |
| `SILENCE_TO_FLUSH` | 0.8 | Silence before an utterance is considered finished. |
| `MAX_BUFFER_SECONDS` | 30 | Hard cap so a long monologue can't grow the buffer unboundedly. |

### Single file

```python
from asr import transcribe
import route_kanji

print(route_kanji.to_romanji(transcribe("test.mp3")))
```

### Reproducing the benchmarks

```powershell
cd "4. mp3Conversion_language_romanji/routes"
python eval_jsut.py        # all 3 routes vs JSUT ground truth -> jsut_results.csv
python eval_readings.py    # all 3 routes vs MeCab reference on Common Voice
```

`eval_jsut.py` scores against verified human kana annotations, so it measures absolute reading
correctness. `eval_readings.py` scores against MeCab's own output on the reference transcript,
so it measures *only* how far the audio pipeline drifts from the text pipeline — MeCab's reading
errors are baked into that ground truth. They answer different questions; `eval_jsut.py` is the
one to trust for accuracy claims.

Clip count is `N_CLIPS` at the top of each file. A 500-clip JSUT run takes roughly 40 minutes on
an RTX-class GPU, since each clip goes through Whisper three times.

## Repo layout

Directories are numbered in the order they were built — each one is a working stage that the
next one builds on.

| Directory | What it does |
|---|---|
| `1. mp3Conversion/` | Minimum viable pipeline: Whisper → pykakasi → rōmaji |
| `2. mp3Conversion_language/` | CER scoring harness on Common Voice; VAD + normalization fixes |
| `3. languageConversion_romanji/` | Text-only bakeoff: pykakasi vs MeCab on hand-written hard readings |
| `4. mp3Conversion_language_romanji/routes/` | The three routes + both evaluation harnesses |
| `5.mp3Conversion_romanji_live/` | Real-time mic pipeline |
| `plan.txt` | Running lab notebook — raw numbers and reasoning, in chronological order |

Inside `routes/`:

| File | Role |
|---|---|
| `asr.py` | Shared Whisper wrapper (file → kanji text) |
| `route_kanji.py` | Whisper → MeCab → rōmaji, with particle sound rules (は→わ, へ→え, を→お) |
| `route_kana.py` | Whisper w/ hiragana prompt → rōmaji |
| `route_hybrid.py` | Kanji route, arbitrated by the kana route via JMdict |
| `homographs.py` | JMdict reading lookup, lazily loaded and LRU-cached |
| `eval_jsut.py` | Scores all routes against JSUT ground truth |
| `eval_readings.py` | Scores all routes against a MeCab reference |

## Design notes

**Particle sounds are orthographic, not phonetic.** The particles は, へ, を are pronounced
*wa*, *e*, *o* — not *ha*, *he*, *wo*. A character-level romanizer gets these wrong on nearly
every sentence. Handling them needs part-of-speech information, which is why `route_kanji.py`
checks `word.feature.pos1 == "助詞"` before substituting, rather than doing a blind string
replace that would corrupt every は inside a normal word.

**Normalize both sides before scoring, and normalize identically.** `canon()` in `eval_jsut.py`
collapses long vowels (`ou`→`oo`, `ei`→`ee`, jaconv's chouon hyphen), applies the を→o particle
convention, and strips everything non-alphabetic. Without this, the benchmark measures
romanization-convention mismatches instead of pronunciation errors — and it was doing exactly
that for a while, which is why an early run reported the hybrid route at 0.929 when the real
number was 0.978.

**Verify your improvement actually fired.** The first hybrid run scored *identically* to the
kanji route across all 200 clips. The apparent conclusion was "no homophones in this sample";
the real cause was a silently missing JMdict database. `eval_jsut.py` now prints
`kanji == hybrid on N/M clips` and warns explicitly when a route produces no distinct output,
because a metric that looks plausible while the code under test never ran is the expensive
kind of wrong.

**Keep the audio callback cheap.** `mic_stream.py` does nothing in `_callback` but copy samples
into a `queue.Queue`. Any real work there — including Whisper — would block the audio thread and
drop samples. All processing happens on the main loop, which drains the queue.

**Different models for different jobs.** Batch evaluation uses `large-v3` for accuracy. The live
pipeline uses `medium` with `beam_size=1` (greedy) because it re-transcribes every second and
`large-v3` can't keep up. It also sets `condition_on_previous_text=False` — with repeated partial
passes over a growing buffer, conditioning causes the model to drift and repeat itself.

## Limitations & next steps

- **VAD is a bare RMS threshold.** `vad.py` is an energy detector; it will treat background noise
  as speech. Swapping in `webrtcvad` (30ms int16 frames) is the obvious upgrade.
- **The kana route is the weak link in the hybrid design.** At 0.878 it's often wrong when it
  overrules MeCab. A dedicated Japanese kana ASR model (ReazonSpeech, or an ESPnet kana model)
  would give the arbitration a reference worth trusting.
- **Remaining hybrid losses are one bug class.** Long-vowel and shortened orthographic variants
  (`hontoo`/`honto`, `akanboo`/`akanbo`) need the same normalization `_skel()` already applies to
  っ and ー.
- **No latency/accuracy curve yet.** `STEP_SECONDS` and buffer size trade responsiveness against
  correctness, and the current values are guesses. Sweeping them and plotting CER against measured
  latency would let the operating point be chosen rather than assumed.
- **No test suite.** `resolve_word()`, `_skel()`, `canon()`, and `common_prefix()` are pure
  functions, and the failure tables above are effectively a ready-made regression corpus.

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 reimplementation of OpenAI Whisper
- [fugashi](https://github.com/polm/fugashi) / [UniDic](https://clrd.ninjal.ac.jp/unidic/) — MeCab morphological analysis
- [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [jsut-label](https://github.com/sarulab-speech/jsut-label) — evaluation audio and kana annotations
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — evaluation audio
- [JMdict / jamdict](https://github.com/neocl/jamdict) — homograph reading lookup
