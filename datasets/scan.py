""" domain knowledge for SCAN
"""
import random
from torch.utils.data import Dataset
from .helper import Program


class SCAN(Dataset):
    # Domain knowledge and rule-based parser for SCAN dataset.
    name = 'scan'
    action_word = ['turn', 'walk', 'look', 'run', 'jump']
    dir_word = ['left', 'right']
    turn_times_word = ['opposite', 'around']
    times_word = ['twice', 'thrice']
    connect_word = ['and', 'after']

    vocab = action_word + dir_word + turn_times_word + times_word + connect_word
    i2w = vocab
    w2i = {w: i for i, w in enumerate(vocab)}

    vocab_output = ['', 'I_WALK', 'I_LOOK', 'I_RUN', 'I_JUMP', 'I_TURN_LEFT', 'I_TURN_RIGHT']
    w2i_output = {w: i for i, w in enumerate(vocab_output)}


    op2precedence = {}
    op2precedence.update({x: 1 for x in connect_word})
    op2precedence.update({x: 2 for x in times_word})
    op2precedence.update({x: 3 for x in turn_times_word})
    op2precedence.update({x: 4 for x in dir_word})

    
    sym2prog = {
        'turn': Program(lambda: ()),
        'walk': Program(lambda: (1,)),
        'look': Program(lambda: (2,)),
        'run': Program(lambda: (3,)),
        'jump': Program(lambda: (4,)),

        'left': Program(lambda x: (5,) + x),
        'right': Program(lambda x: (6,) + x),

        'opposite': Program(lambda x: (x[0],) + x),
        'around': Program(lambda x: x * 4),

        'twice': Program(lambda x: x * 2),
        'thrice': Program(lambda x: x * 3),

        'and': Program(lambda x, y: x + y),
        'after': Program(lambda x, y: y + x),
    }
    
    sym2arity = {k: v.arity for k, v in sym2prog.items()}

    sym2learnable = {x: True for x in vocab}
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
            if sym2arity[sym] == 0:
                values.append(i)
            else:
                while len(operators) > 0 and op2precedence[expr[operators[-1]]] >= op2precedence[sym]:
                    op = operators.pop()
                    for _ in range(sym2arity[expr[op]]):
                        head[values.pop()] = op
                    values.append(op)
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
        with open(filename, 'r') as f:
            lines = f.readlines()
        dataset = []
        for line in lines:
            _, left, right = line.split(':')
            left = left.strip().split()[:-1]
            left = cls.transform_expr(left)
            head = cls.parse(left)
            right = right.strip().split()
            data = {'expr': left, 'head': head, 'res': right}
            dataset.append(data)
        return dataset
    
    @classmethod
    def transform_expr(cls, expr):
        """ 
        opposite/around left/right -> left/right opposite/around
        this transform makes SCAN's dependency grammar projective and can be parsed by a shift-reduce parser.
        """
        for i, w in enumerate(expr[:]):
            if w in cls.turn_times_word:
                expr[i] = expr[i+1]
                expr[i+1] = w
        return expr

    def __init__(self, split='train', name='length', n_sample=None):

        assert split in ['train', 'val', 'test']
        split = 'test' if split == 'val' else split # there is no val split for scan
        if name in ['simple', 'length']:
            filename = f'./datasets/scan/{name}_split/tasks_{split}_{name}.txt'
        elif name == 'addprim_jump':
            filename = f'./datasets/scan/add_prim_split/tasks_{split}_{name}.txt'
        elif name == 'addprim_turn_left':
            filename = f'./datasets/scan/add_prim_split/tasks_{split}_{name}.txt'
        elif name == 'template_around_right':
            filename = f'./datasets/scan/template_split/tasks_{split}_{name}.txt'
        else:
            assert False, f'Unknown split for SCAN: {name}'
        
        dataset = self.load_data(filename)

        if n_sample:
            if n_sample <= 1: # it is percentage
                n_sample = int(len(dataset) * n_sample)
            random.shuffle(dataset)
            dataset = dataset[:n_sample]
            print(f'{split}: randomly select {n_sample} samples.')

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
    dataset = SCAN()
    print(dataset[0])
    pass

