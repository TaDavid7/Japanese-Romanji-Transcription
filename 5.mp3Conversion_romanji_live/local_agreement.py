# LocalAgreement-2: only trust text that survived two consecutive Whisper runs.
def common_prefix(a, b):
    out = []
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        out.append(ca)
    return "".join(out)

class LocalAgreement:
    def __init__(self):
        self.prev = ""       # last run's hypothesis
        self.committed = ""  # text we've locked in

    def update(self, hypothesis):
        # newly agreed prefix = what these two runs share, beyond what's committed
        agreed = common_prefix(self.prev, hypothesis)
        new_text = agreed[len(self.committed):] if len(agreed) > len(self.committed) else ""
        self.committed = agreed
        self.prev = hypothesis
        return new_text  # the freshly-confirmed characters (may be "")

    def flush(self):
        # commit whatever's left, reset for next utterance
        tail = self.prev[len(self.committed):]
        self.prev, self.committed = "", ""
        return tail