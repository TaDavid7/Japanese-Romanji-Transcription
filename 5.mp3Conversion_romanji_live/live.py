# Main live loop: Mic -> ring buffer -> VAD -> Whisper -> LocalAgreement -> romanji
import time
import numpy as np
from mic_stream import MicStream, SAMPLE_RATE
from vad import is_speech
from local_agreement import LocalAgreement
from asr_live import transcribe_array
from to_romanji import to_romanji

STEP_SECONDS = 1.0        # re-run Whisper this often
SILENCE_TO_FLUSH = 0.8    # seconds of quiet = end of utterance
MAX_BUFFER_SECONDS = 30   # safety cap so the buffer can't grow forever

def main():
    buffer = np.zeros(0, dtype=np.float32)
    agree = LocalAgreement()
    silence_time = 0.0
    last_run = time.time()

    with MicStream() as mic:
        print("Listening... (Ctrl+C to stop)\n")
        while True:
            block = mic.read()
            if block is not None:
                buffer = np.concatenate([buffer, block])
                # track silence for end-of-utterance detection
                silence_time = 0.0 if is_speech(block) else silence_time + len(block) / SAMPLE_RATE

            # every STEP_SECONDS, transcribe the growing buffer
            if time.time() - last_run >= STEP_SECONDS and len(buffer) > SAMPLE_RATE:
                last_run = time.time()
                hyp = transcribe_array(buffer)
                new_text = agree.update(hyp)
                if new_text:
                    print(to_romanji(new_text), end=" ", flush=True)

            # end of utterance: flush the pending tail, drop the buffer, reset
            if silence_time >= SILENCE_TO_FLUSH and len(buffer) > 0:
                tail = agree.flush()
                if tail:
                    print(to_romanji(tail))
                print()  # newline between utterances
                buffer = np.zeros(0, dtype=np.float32)
                silence_time = 0.0

            # hard cap so a long monologue doesn't blow up latency
            if len(buffer) > MAX_BUFFER_SECONDS * SAMPLE_RATE:
                agree.flush()
                buffer = np.zeros(0, dtype=np.float32)

            time.sleep(0.05)

if __name__ == "__main__":
    main()