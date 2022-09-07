from .semantics import DreamCoder, Semantics

class SemanticsGT():
    def __init__(self, config):
        self.semantics = [Semantics(i, arity=config.domain.sym2arity[s], program=config.domain.sym2prog[s], learnable=False) 
                    for i, s in enumerate(config.domain.i2w)]

    def __call__(self):
        return self.semantics

    def save(self):
        pass

    def load(self, model):
        pass


def build(config=None):
    if config.semantics:
        model = SemanticsGT(config)
    else:
        model = DreamCoder(config)
    return model