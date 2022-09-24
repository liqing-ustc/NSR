
from .perception import Perception

class NULLPerception:
    def __init__(self):
        self.training = False
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def save(self):
        pass

    def load(self, *args):
        pass

    def to(self, *args):
        pass

def build(config):
    model = Perception(config.domain) if not config.perception else NULLPerception()
    return model
