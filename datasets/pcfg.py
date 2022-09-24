""" domain knowledge for PCFG
"""
import random
from torch.utils.data import Dataset
from .helper import Program

class PCFG(Dataset):
    # Domain knowledge and rule-based parser for PCFG dataset.
    name = 'pcfg'
    unary_functions = ['copy', 'reverse', 'shift', 'echo', 'swap_first_last', 'repeat']
    binary_functions = ['append', 'prepend', 'remove_first', 'remove_second']
    separator = ','
    arguments = [str(i) for i in range(47)] # the maximum number of arguments in an example is 47

    vocab = arguments + unary_functions + binary_functions + [separator]
    i2w = vocab
    w2i = {w: i for i, w in enumerate(vocab)}

    vocab_output = arguments
    w2i_output = {w: i for i, w in enumerate(vocab_output)}

    op2precedence = {}
    op2precedence.update({x: 1 for x in vocab})

    sym2prog = {
        # see definitions from https://github.com/MathijsMul/pcfg-set/blob/master/tasks/default.py
        'copy': lambda x: x,
        'reverse': lambda x: x[::-1],
        'shift': lambda x: x[1:] + x[:1],
        'echo': lambda x: x + x[-1:],
        'swap_first_last': lambda x: x[-1:] + x[1:-1] + x[:1],
        # append (cons (car (reverse $0) reverse(cdr (reverse (cdr $0)))) singleton (car $0))
        'repeat': lambda x: x + x,
        'append': lambda x,y: x + y,
        'prepend': lambda x,y: y + x,
        'remove_first': lambda x,y: y,
        'remove_second': lambda x,y: x,
    }
    sym2prog[separator] = lambda: ()
    # how to create a list of functions by for loop: https://stackoverflow.com/a/2295368
    sym2prog.update({k: (lambda y: lambda x: (y,) + x)(i)  for k, i in w2i_output.items()})
    sym2prog = {k: Program(v) for k, v in sym2prog.items()}

    sym2arity = {k: v.arity for k, v in sym2prog.items()}

    sym2learnable = {separator: True}
    sym2learnable.update({x: True for x in arguments})
    sym2learnable.update({x: True for x in unary_functions})
    sym2learnable.update({x: True for x in binary_functions})

    curriculum = dict([
            (0, 10),
            (20, 20),
            (40, float('inf')),
    ])


    @classmethod
    def parse(cls, expr):
        sym2arity = cls.sym2arity
        op2precedence = cls.op2precedence
        values = []
        operators = []
        
        head = [-1] * len(expr)
        for (i,sym) in enumerate(expr):
            operators.append(i)

        while len(operators) > 0:
            op = operators.pop()
            for _ in range(sym2arity[expr[op]]):
                head[values.pop()] = op
            values.append(op)

        root_op = values.pop()
        head[root_op] = -1
        assert len(values) == 0

        return head
    
    @classmethod
    def load_data(cls, filename):
        src_file = filename + '.src'
        tgt_file = filename + '.tgt'
        with open(src_file, 'r') as f:
            src_lines = f.readlines()
        with open(tgt_file, 'r') as f:
            tgt_lines = f.readlines()
        assert len(src_lines) == len(tgt_lines)
        dataset = []
        for left, right in zip(src_lines, tgt_lines):
            left = left.strip().split()
            right = right.strip().split()
            left, right = cls.transform_expr(left, right)
            head = cls.parse(left)
            data = {'expr': left, 'head': head, 'res': right}
            dataset.append(data)
        return dataset
    
    @classmethod
    def primitive_data(cls, n_args=5, n_example=1000):
        primitives = [tuple(random.choices(cls.arguments, k=i)) for i in range(n_args) for _ in range(n_example)]

        dataset = []
        for prim in set(primitives):
            prim = list(prim)
            left = prim + [',']
            right = prim
            head = cls.parse(left)
            data = {'expr': left, 'head': head, 'res': right}
            dataset.append(data)
        
        return dataset

    
    @classmethod
    def transform_expr(cls, left, right):
        """ 
        (1) replace the arguments into index. This doesn't affect the performance, but reduce the vocab size.
        (2) add a separator to the end of input. This makes the arity of arguments to 1.
        E.g.,  input: 'copy R11 B10'  output: 'R11 B10' -> input: 'copy 1 0 ,'  output: '1 0'
        """
        args_left = []
        for x in left:
            if x in args_left or x in cls.unary_functions + cls.binary_functions + [cls.separator]:
                continue
            args_left.append(x)
        # args_left = [x for x in left if x not in cls.unary_functions + cls.binary_functions + [cls.separator]]
        # args_left = sorted(list(set(args_left)))
        mapping = {x: str(i) for i, x in enumerate(args_left)}
        left = [mapping.get(x, x) for x in left] + [cls.separator]
        right = [mapping[x] for x in right]
        return left, right

    def __init__(self, split='train', name='pcfgset', n_sample=None):

        if name == 'pcfgset':
            if split == 'val':
                split = 'dev'
            filename = f'./datasets/pcfg/{name}/{split}'
        elif name in ['systematicity', 'productivity']:
            if split == 'val':
                split = 'test'
            filename = f'./datasets/pcfg/{name}/{split}'
        
        dataset = self.load_data(filename)

        if n_sample:
            if n_sample <= 1: # it is percentage
                n_sample = int(len(dataset) * n_sample)
            random.shuffle(dataset)
            dataset = dataset[:n_sample]
            print(f'{split}: randomly select {n_sample} samples.')

        if split == 'train':
            dataset = self.primitive_data() + dataset

        for sample in dataset:
            sample['len'] = len(sample['expr'])
            sample['sentence'] = [self.w2i[x] for x in sample['expr']]
            sample['res'] = tuple([self.w2i_output[x] for x in sample['res']])
        
        self.dataset = dataset
        self.valid_ids = list(range(len(dataset)))

    def __getitem__(self, index):
        index = self.valid_ids[index]
        sample = self.dataset[index]
        
        return sample
    
    def __len__(self):
        return len(self.valid_ids)

    def filter_by_len(self, min_len=None, max_len=None):
        if min_len is None: min_len = -1
        if max_len is None: max_len = float('inf')
        self.valid_ids = [i for i, x in enumerate(self.dataset) if x['len'] <= max_len and x['len'] >= min_len]
    
    
    @classmethod
    def collate(cls, batch):
        expr_list = []
        sentence_list = []
        head_list = []
        res_list = []
        len_list = []
        for sample in batch:
            expr_list.append(sample['expr'])
            sentence_list.append(sample['sentence'])
            head_list.append(sample['head'])
            res_list.append(sample['res'])
            len_list.append(sample['len'])
            
        batch = {}
        batch['expr'] = expr_list
        batch['sentence'] = sentence_list
        batch['head'] = head_list
        batch['res'] = res_list
        batch['len'] = len_list
        return batch


if __name__ == '__main__':
    dataset = PCFG()
    print(dataset[0])
    pass

