from inspect import signature

START = '<START>'
END = '<END>'
NULL = '<NULL>'

EMPTY_VALUE = -1
MISSING_VALUE = -2

class Program():
    def __init__(self, fn=None):
        self.fn = fn
        self.arity = len(signature(fn).parameters) if fn is not None else 0
        self.likelihood = 1.0
        self.cache = {} # used for fast computation
        self.prog = 'GT'

    def __call__(self, inputs):
        if len(inputs) != self.arity or MISSING_VALUE in inputs:
            return MISSING_VALUE
        try:
            y = self.fn(*inputs)
        except (TypeError, RecursionError, IndexError) as e:
            print(repr(e))
            y = MISSING_VALUE
        self.cache[inputs] = y
        return y

    def evaluate(self, examples, store_y=True): 
        ys = []
        for exp in examples:
            y = self(exp)
            ys.append(y)
        return ys

    def __str__(self):
        return str(self.prog)

    def solve(self, i, inputs, output_list):
        if len(inputs) != self.arity:
            return []
        
        def equal(a, b, pos):
            for j in range(len(a)):
                if j == pos:
                    continue
                if a[j] != b[j]:
                    return False
            return True

        candidates = []
        for xs, y in self.cache.items():
            if y in output_list and equal(xs, inputs, i):
                candidates.append(xs[i])
        return candidates