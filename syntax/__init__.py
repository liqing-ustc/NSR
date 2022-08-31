from .parser import Parser, PartialParse

def build(config):
    model = Parser()
    return model

def convert_trans2dep(transitions):
    s_len = (len(transitions) + 1)//2
    parse = PartialParse(list(range(s_len)))
    parse.parse(transitions)
    return parse.dependencies