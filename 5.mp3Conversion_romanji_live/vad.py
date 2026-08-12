# VAD: is this block speech or silence? Simple energy threshold to start.
import numpy as np

def is_speech(block, threshold=0.01):
    # RMS loudness of the block; above threshold = someone's talking
    rms = np.sqrt(np.mean(block ** 2))
    return rms > threshold
#Better later: pip install webrtcvad, feed it 30 ms frames of int16 audio. The energy version is enough to prove the loop works.