from .parser import Parser, PartialParse

def build(config):
    n_tokens = len(config.domain.i2w)
    i2arity = [config.domain.sym2arity[config.domain.i2w[i]] for i in range(n_tokens)] \
        if getattr(config.domain, 'enforce_arity_parsing', False) else None
    model = Parser(n_tokens, i2arity)
    return model

def convert_trans2dep(transitions):
    s_len = (len(transitions) + 1)//2
    parse = PartialParse(list(range(s_len)))
    parse.parse(transitions)
    return parse.dependencies