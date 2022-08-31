from utils import SYMBOLS, SYM2PROG
from .semantics import DreamCoder, Semantics

class SemanticsGT():
    def __init__(self):
        self.semantics = [Semantics(i, SYM2PROG[s], learnable=False) for i, s in enumerate(SYMBOLS)]

    def __call__(self):
        return self.semantics

    def save(self):
        pass

    def load(self, model):
        pass


def build(config=None):
    if config.semantics:
        model = SemanticsGT()
    else:
        model = DreamCoder(config)
    return model