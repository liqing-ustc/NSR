""" domain knowledge for HINT
"""
import random, json, math
from copy import deepcopy
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .helper import Program, MISSING_VALUE, EMPTY_VALUE


class HINT(Dataset):
    name = 'hint'
    operators = ['+', '-', '*', '/']
    parentheses = ['(', ')']
    digits = [str(i) for i in range(10)]

    vocab = digits + operators + parentheses
    i2w = vocab
    w2i = {w: i for i, w in enumerate(vocab)}

    vocab_output = digits
    w2i_output = {w: i for i, w in enumerate(vocab_output)}

    op2precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    sym2prog = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: max(0, x - y),
        '*': lambda x, y: x * y,
        '/': lambda x, y: math.ceil(x / y) if y != 0 else MISSING_VALUE,
        '(': lambda: EMPTY_VALUE, ')': lambda: EMPTY_VALUE
    }
    # how to create a list of functions by for loop: https://stackoverflow.com/a/2295368
    sym2prog.update({i: (lambda x: lambda: x)(int(i))  for i in digits})
    sym2prog = {k: Program(v) for k, v in sym2prog.items()}

    sym2arity = {k: v.arity for k, v in sym2prog.items()}
    sym2learnable = {x: True for x in vocab}
    sym2learnable.update({x: False for x in parentheses}) # do not learn semantics for parentheses

    curriculum = dict([
            (0, 3),
            (20, 7),
            (40, 11),
            (60, 15),
            (80, float('inf')),
    ])

    img_size = 32
    img_transform = transforms.Compose([
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: 1. - x),
                    # transforms.Normalize((0.5,), (1,))
                ])

    root_dir = './datasets/hint/'
    img_dir = root_dir + 'symbol_images/'
    perception_pretrain = root_dir + 'perception_pretrain/model.pth.tar_78.2_match'
    update_grammar = True # whether to update grammar when learning semantics
    enforce_arity_parsing = False # do not enforce the arity constraint for parsing because parentheses has no output.


    @classmethod
    def parse(cls, expr):
        sym2arity = cls.sym2arity
        op2precedence = cls.op2precedence
        lps, rps = cls.parentheses
        values = []
        operators = []
        
        head = [-1] * len(expr)
        for (i,sym) in enumerate(expr):
            if sym == lps:
                operators.append(i)
            elif sym == rps:
                while expr[operators[-1]] != lps:
                    op = operators.pop()
                    for _ in range(sym2arity[expr[op]]):
                        head[values.pop()] = op
                    values.append(op)
                i_lps = operators[-1]
                i_rps = i
                head[i_lps] = op
                head[i_rps] = op
                operators.pop()
            elif sym2arity[sym] == 0:
                values.append(i)
            else:
                while len(operators) > 0 and expr[operators[-1]] != lps and \
                    op2precedence[expr[operators[-1]]] >= op2precedence[sym]:
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
    def expr2n_op(cls, expr):
        return len([1 for x in expr if x in cls.operators])

    @classmethod
    def load_image(cls, img_path):
        def pad_image(img, desired_size, fill=0):
            delta_w = desired_size - img.size[0]
            delta_h = desired_size - img.size[1]
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            new_img = ImageOps.expand(img, padding, fill)
            return new_img
        img = Image.open(cls.img_dir + img_path).convert('L')
        img = ImageOps.invert(img)
        img = pad_image(img, 60)
        img = transforms.functional.resize(img, 40)
        img = cls.img_transform(img)
        return img

    @classmethod
    def render_img(cls, img_paths):
        images = [Image.open(cls.img_dir + x) for x in img_paths]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('L', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        return new_im

    def __init__(self, split='train', name='image', n_sample=None, fewshot=None, max_op=None, main_dataset_ratio=0.):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.input = name
        self.fewshot = fewshot

        root_dir = self.root_dir
        if fewshot:
            dataset = json.load(open(root_dir + 'fewshot_dataset.json'))
            dataset = dataset[fewshot]
            dataset = dataset[split]
            self.main_dataset_ratio = main_dataset_ratio
            if split == 'train' and main_dataset_ratio > 0:
                self.main_dataset = json.load(open(root_dir + 'expr_%s.json'%split))
        else:
            dataset = json.load(open(root_dir + 'expr_%s.json'%split))

        if n_sample:
            if n_sample <= 1: # it is percentage
                n_sample = int(len(dataset) * n_sample)
            random.shuffle(dataset)
            dataset = dataset[:n_sample]
            print(f'{split}: randomly select {n_sample} samples.')
            
        if isinstance(max_op, int):
            dataset = [x for x in dataset if self.expr2n_op(x['expr']) <= max_op]
            print(f'{split}: filter {len(dataset)} samples with no more than {max_op} operators.')

        for sample in dataset:
            sample['len'] = len(sample['expr'])
            sample['sentence'] = [self.w2i[x] for x in sample['expr']]
        
        self.dataset = dataset
        self.valid_ids = list(range(len(dataset)))

    def __getitem__(self, index):
        if self.fewshot and self.split == 'train' and random.random() < self.main_dataset_ratio:
            # use sample from main dataset to avoid forgetting
            sample = random.choice(self.main_dataset)
            sample = deepcopy(sample)
        else:
            index = self.valid_ids[index]
            sample = deepcopy(self.dataset[index])
        if self.input == 'image':
            img_seq = []
            for img_path in sample['img_paths']:
                img_seq.append(self.load_image(img_path))

            sample['img_seq'] = img_seq
            sample['len'] = len(img_seq)
        # del sample['img_paths']
        sample['expr'] = ''.join(sample['expr'])
        return sample
    
    def __len__(self):
        return len(self.valid_ids)

    def filter_by_len(self, min_len=None, max_len=None):
        if min_len is None: min_len = -1
        if max_len is None: max_len = float('inf')
        self.valid_ids = [i for i, x in enumerate(self.dataset) if x['len'] <= max_len and x['len'] >= min_len]
    

    def all_exprs(self, max_len=float('inf')):
        dataset = random.sample(self.dataset, min(int(1e4), len(self.dataset)))
        dataset = [sample for sample in dataset if len(sample['expr']) <= max_len]
        return dataset
    
    @classmethod
    def collate(cls, batch):
        img_seq_list = []
        expr_list = []
        sentence_list = []
        img_paths_list = []
        head_list = []
        res_all_list = []
        res_list = []
        len_list = []
        for sample in batch:
            if 'img_seq' in sample:
                img_seq_list.extend(sample['img_seq'])

            expr_list.append(sample['expr'])
            sentence_list.append(sample['sentence'])
            img_paths_list.append(sample['img_paths'])
            head_list.append(sample['head'])
            res_all_list.append(sample['res_all'])
            res_list.append(sample['res'])
            len_list.append(sample['len'])
            
        batch = {}
        if img_seq_list:
            batch['img_seq'] = torch.stack(img_seq_list)
        batch['input'] = img_paths_list
        batch['expr'] = expr_list
        batch['sentence'] = sentence_list
        batch['head'] = head_list
        batch['res_all'] = res_all_list
        batch['res'] = res_list
        batch['len'] = len_list
        return batch