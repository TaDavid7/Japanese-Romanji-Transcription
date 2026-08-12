# Mic -> ring buffer. Grabs 16kHz mono audio off the default input device.
import queue
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper wants 16kHz

class MicStream:
    def __init__(self, samplerate=SAMPLE_RATE, blocksize=1600):  # 1600 = 0.1s blocks
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.q = queue.Queue()

    def _callback(self, indata, frames, time, status):
        # runs on audio thread - keep it cheap, just copy into the queue
        self.q.put(indata[:, 0].copy())  # [:,0] = first channel (mono)

    def __enter__(self):
        self.stream = sd.InputStream(
            samplerate=self.samplerate, channels=1,
            blocksize=self.blocksize, dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop()
        self.stream.close()

    def read(self):
        # drain everything currently queued, return as one array (or None)
        chunks = []
        while not self.q.empty():
            chunks.append(self.q.get())
        return np.concatenate(chunks) if chunks else None