# Main live loop: Mic -> ring buffer -> VAD -> Whisper -> LocalAgreement -> romanji
import time, shutil, sys
import numpy as np
from mic_stream import MicStream, SAMPLE_RATE
from vad import is_speech
from local_agreement import LocalAgreement
from asr_live import transcribe_array
from to_romanji import to_romanji

STEP_SECONDS = 1.0        # re-run Whisper this often
SILENCE_TO_FLUSH = 0.8    # seconds of quiet = end of utterance
MAX_BUFFER_SECONDS = 30   # safety cap so the buffer can't grow forever

def width():
    return shutil.get_terminal_size((80, 20)).columns - 1

def render(line):
    # rewrite the current utterance in place; pad to wipe the previous, longer text
    sys.stdout.write("\r" + line[-width():].ljust(width()))
    sys.stdout.flush()

def finish(line):
    # clear the live line, then print the finished text in full (wraps normally)
    sys.stdout.write("\r" + " " * width() + "\r" + line + "\n")
    sys.stdout.flush()

def main():
    buffer = np.zeros(0, dtype=np.float32)
    agree = LocalAgreement()
    silence_time = 0.0
    last_run = time.time()
    line = ""                 # romanji committed so far this utterance

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
                new_text = agree.update(transcribe_array(buffer))
                if new_text:
                    line = (line + " " + to_romanji(new_text)).strip()   # same as original
                    render(line)

            # end of utterance: flush the pending tail, drop the buffer, reset
            if silence_time >= SILENCE_TO_FLUSH and len(buffer) > 0:
                tail = agree.flush()
                if tail:
                    line = (line + " " + to_romanji(tail)).strip()
                if line:                  # only break the line if something was said
                    finish(line)
                    line = ""
                buffer = np.zeros(0, dtype=np.float32)
                silence_time = 0.0

            # hard cap so a long monologue doesn't blow up latency
            if len(buffer) > MAX_BUFFER_SECONDS * SAMPLE_RATE:
                tail = agree.flush()
                if tail:
                    line = (line + " " + to_romanji(tail)).strip()
                if line:
                    finish(line)
                    line = ""
                buffer = np.zeros(0, dtype=np.float32)

            time.sleep(0.05)

if __name__ == "__main__":
    main()