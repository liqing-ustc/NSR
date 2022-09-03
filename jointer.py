from numpy.core.records import array
import perception, syntax, semantics
import numpy as np
from copy import deepcopy
import sys
from func_timeout import func_timeout, FunctionTimedOut
from utils import DEVICE, SYMBOLS, EMPTY_VALUE, MISSING_VALUE
from collections import Counter, namedtuple
from time import time
import torch
from torch.distributions.categorical import Categorical
import random
from heapq import heappush, heappop, heapify
import queue as Q
from dataclasses import dataclass, field
from typing import Any
@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

Parse = namedtuple('Parse', ['sentence', 'head'])

class SentGenerator(object):
    def __init__(self, probs, training=False):
        probs = np.log(np.maximum(probs, 1e-7))
        self.probs = probs
        self.max_probs = probs.max(1)
        self.queue = [(-self.max_probs.sum(), [])]
        self.training = training
    
    def next(self):
        if self.training:
            m = Categorical(logits=torch.from_numpy(self.probs))
            sent = list(m.sample().numpy())
            return sent

        epsilon = np.log(1e-5)
        while self.queue:
            priority, sent = heappop(self.queue)
            if len(sent) == len(self.probs):
                return sent
            next_pos = len(sent)
            next_prob = self.probs[next_pos]
            for i, p in enumerate(next_prob):
                if p < epsilon:
                    continue
                new_state = (priority + self.max_probs[next_pos] - p, sent + [i])
                heappush(self.queue, new_state)
        
        return None

class Node:
    def __init__(self, index, symbol, smt, prob=0.):
        self.index = index
        self.symbol = symbol
        self.smt = smt
        self.children = []
        self.sym_prob = prob
        self._res = MISSING_VALUE
        self._res_computed = False

    def res(self):
        if self._res_computed:
            return self._res

        res = self.smt(self.inputs())
        if isinstance(res, int) and res > sys.maxsize:
            res = MISSING_VALUE
        self.prob = self.sym_prob + np.log(max(self.smt.likelihood, 1e-7)) + sum([x.prob for x in self.children])
        self._res = res
        self._res_computed = True
        return self._res

    def inputs(self):
        return [x.res() for x in self.children]
    
    def copy(self, node):
        self.index = node.index
        self.symbol = node.symbol
        self.children = node.children
        self.smt = node.smt
        self.prob = node.prob
        self._res = node._res
        self._res_computed = node._res_computed

class AST: # Abstract Syntax Tree
    def __init__(self, pt, semantics, sent_probs=None):
        self.pt = pt
        self.semantics = semantics
        self.sent_probs = np.log(np.maximum(sent_probs, 1e-7))

        nodes = [Node(i, s, semantics[s], self.sent_probs[i][s]) for i, s in enumerate(pt.sentence)]

        for node, h in zip(nodes, pt.head):
            if h == -1:
                self.root_node = node
                continue
            nodes[h].children.append(node)
        self.nodes = nodes

        self.root_node.res() 

    def res(self): return self.root_node.res()

    def res_all(self): return [nd._res for nd in self.nodes]

    def abduce(self, y, module=None):
        queue = Q.PriorityQueue()
        change = PrioritizedItem(0., (self.root_node, [y]))
        queue.put(change)
        find_fix = False
        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target = change.item
            if node.res() in target:
                find_fix = True
                break
            
            if node not in self.nodes:
                # this node is generated when abducing, no need to abduce it.
                continue

            changes = []
            if module == 'perception':
                changes.extend(self.abduce_perception(node, target))
            # if module == 'syntax':
            #     changes.extend(self.abduce_syntax(node, target))
            if module == 'semantics':
                changes.extend(self.abduce_semantics(node, target))
            
            changes.extend(self.top_down(node, target))

            for x in changes:
                queue.put(x)
        
        if find_fix:
            if module == 'syntax':
                if hasattr(node, 'head'):
                    self.pt.head = node.head
                return self

            original_node = self.nodes[node.index]
            original_node.copy(node)
            
            def ancestors(i):
                h = self.pt.head[i]
                if h == -1:
                    return []
                return ancestors(h) + [h]
            
            for i in ancestors(node.index):
                self.nodes[i]._res_computed = False

            self.root_node.res()
            self.pt.sentence = [node.symbol for node in self.nodes]
            return self
        return None

    def top_down(self, node, target):
        changes = []
        inputs = node.inputs()
        for pos, ch in enumerate(node.children):
            inputs_valid = [x for i, x in enumerate(inputs) if x != EMPTY_VALUE or i == pos]
            pos -= inputs[:pos].count(EMPTY_VALUE)
            ch_target = []
            if len(ch.children) == 0 and ch.res() != EMPTY_VALUE: # when ch has no children, test if its output can be EMPTY_VALUE
                new_inputs = inputs_valid[:]
                del new_inputs[pos]
                if node.smt(new_inputs) in target:
                    ch_target.append(EMPTY_VALUE)
            ch_target.extend(node.smt.solve(pos, inputs_valid, target))
            if ch_target:
                priority = ch.prob - np.log(np.maximum(1. - np.exp(ch.prob), 1e-7))
                changes.append(PrioritizedItem(priority, (ch, ch_target)))
        return changes

    def abduce_perception(self, node, target):
        changes = []
        probs = self.sent_probs[node.index]
        for sym in range(len(SYMBOLS)):
            if sym == node.symbol:
                continue
            smt = self.semantics[sym]
            new_node = Node(node.index, sym, smt, probs[sym])
            new_node.children = node.children
            priority = probs[node.symbol] - probs[sym]
            changes.append(PrioritizedItem(priority, (new_node, target)))
        return changes

    def abduce_syntax(self, node, target):
        arcs = self.pt.dependencies
        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k], reverse=True)

        changes = []
        for ch in node.children:
            # rotate the arc between node and ch, and create new head
            h = node.index
            t = ch.index
            for arc in arcs:
                if arc[0] == h and arc[1] == t:
                    p = arc[2]
                    break
            head = self.pt.head[:]

            head[t] = head[h]
            head[h] = t 

            children = get_rc(h) if h < t else get_lc(h)
            for j in children[:children.index(t)]:
                head[j] = t

            children = get_lc(t) if h < t else get_rc(t)
            for j in children:
                head[j] = h
            
            # create a new head node from ch and a tail node from the original node
            # update their children according to the new head
            head_node = Node(0,0,0)
            head_node.copy(ch)
            head_node._res_computed = False
            head_node.children = []
            head_node.head = head # store the head

            tail_node = Node(0,0,0)
            tail_node.copy(node)
            tail_node._res_computed = False
            tail_node.children = []
            for t, h in enumerate(head):
                if h == tail_node.index:
                    assert t != head_node.index
                    tail_node.children.append(self.nodes[t])
                elif h == head_node.index:
                    if t == tail_node.index:
                        head_node.children.append(tail_node)
                    else:
                        head_node.children.append(self.nodes[t])

            priority = np.log(np.maximum(p, 1e-7)) - np.log(np.maximum(1 - p, 1e-7))
            changes.append(PrioritizedItem(priority, (head_node, target)))

        return changes

    def abduce_semantics(self, node, target):
        inputs = node.inputs()
        if EMPTY_VALUE in target and len(inputs) > 0:
            return []
        if len(inputs) > 0 and (MISSING_VALUE in inputs):
            return []

        new_node = Node(0,0,0)
        new_node.copy(node)
        new_node._res = random.choice(target)
        priority = np.log(max(node.smt.likelihood, 1e-7)) - np.log(max(1 - node.smt.likelihood, 1e-7))
        return [PrioritizedItem(priority, (new_node, target))]
    
class Jointer:
    def __init__(self, config=None):
        super(Jointer, self).__init__()
        self.config = config
        self.perception = perception.build(config)
        self.syntax = syntax.build(config)
        self.semantics = semantics.build(config)
        self.ASTs = []
        self.buffer = []
        self.epoch = 0
        self.learning_schedule = ['semantics'] * (0 if config.semantics else 1) \
                               + ['perception'] * (0 if config.perception else 1) \
                               + ['syntax'] * (0 if config.syntax else 1) \

    @property
    def learned_module(self):
        return self.learning_schedule[self.epoch % len(self.learning_schedule)]

    def save(self, save_path, epoch=None):
        model = {'epoch': epoch}
        model['perception'] = self.perception.save()
        model['syntax'] = self.syntax.save()
        model['semantics'] = self.semantics.save()
        torch.save(model, save_path)

    def load(self, load_path):
        model = torch.load(load_path)
        self.perception.load(model['perception'])
        self.syntax.load(model['syntax'])
        self.semantics.load(model['semantics'])
        return model['epoch']

    def extend(self, n=1): # extend n new concepts
        self.perception.extend(n)
        self.syntax.extend(n)
        self.semantics.extend(n)

    def print(self):
        if self.config.perception:
            print('use ground-truth perception.')
        else:
            print(self.perception.model)
        if self.config.syntax:
            print('use ground-truth syntax.')
        else:
            print(self.syntax.model)
        if self.config.semantics:
            print('use ground-truth semantics.')
        else:
            self.semantics._print_semantics()

    def train(self):
        self.perception.train()
        self.syntax.train()
        # self.semantics.train()
    
    def eval(self):
        self.perception.eval()
        self.syntax.eval()
        # self.semantics.eval()

    def to(self, device):
        self.perception.to(device)
        self.syntax.to(device)
    
    def deduce(self, sample, n_steps=1):
        config = self.config
        lengths = sample['len']

        if config.perception: # use gt perception
            sentences = sample['sentence']
            sent_probs = []
            for sent, l in zip(sentences, lengths):
                probs = np.zeros((l, len(SYMBOLS)))
                probs[range(l), sent] = 1
                sent_probs.append(probs)
        else:
            pass

        semantics = self.semantics()
        self.ASTs = [None] * len(lengths)
        sent_generators = [SentGenerator(probs, self.perception.training) for probs in sent_probs]
        unfinished = list(range(len(lengths)))
        for t in range(n_steps):
            sentences = [sent_generators[i].next() for i in unfinished]
            not_none = [i for i, s in enumerate(sentences) if s is not None]
            unfinished = [unfinished[i] for i in not_none]
            sentences = [sentences[i] for i in not_none]
            if config.syntax: # use gt parse
                parses = []
                for i, s in zip(unfinished, sentences):
                    head = sample['head'][i]
                    pt = syntax.PartialParse(s)
                    pt.head = head
                    parses.append(pt)
            else:
                parses = self.syntax(sentences)
            
            tmp = []
            for i, pt in zip(unfinished, parses):
                ast = AST(pt, semantics, sent_probs[i])
                if ast.res() is MISSING_VALUE or ast.res() is EMPTY_VALUE:
                    tmp.append(i)
                if self.ASTs[i] is None or ast.res() is not MISSING_VALUE:
                    self.ASTs[i] = ast
            unfinished = tmp
            if not unfinished:
                break

        results = [ast.res() for ast in self.ASTs]
        head = [ast.pt.head for ast in self.ASTs]
        sentences = [ast.pt.sentence for ast in self.ASTs]
        return results, sentences, head

    def abduce(self, gt_values):
        for et, y in zip(self.ASTs, gt_values):
            new_et = et.abduce(y, self.learned_module)
            if new_et: 
                self.buffer.append(new_et)
    
    def clear_buffer(self):
        self.buffer = []

    def learn(self):
        if len(self.buffer) == 0: 
            return

        self.train()
        print("Hit samples: ", len(self.buffer), ' Ave length: ', round(np.mean([len(x.pt.sentence) for x in self.buffer]), 2))
        pred_symbols = Counter([y for x in self.buffer for y in x.pt.sentence])
        print("Symbols: ", len(pred_symbols), sorted(pred_symbols.items()))
        pred_heads = Counter([tuple(ast.pt.head) for ast in self.buffer])
        print("Head: ", sorted(pred_heads.most_common(10), key=lambda x: len(x[0])))

        if self.config.fewshot:
            self.buffer = self.buffer + random.sample(self.buffer_augment, k=1000)

        if self.learned_module == 'perception':
            dataset = [(x.img_paths, x.pt.sentence) for x in self.buffer]
            n_iters = int(100)
            print("Learn perception with %d samples for %d iterations, "%(len(self.buffer), n_iters), end='', flush=True)
            st = time()
            self.perception.learn(dataset, n_iters=n_iters)
            print("take %d sec."%(time()-st))

        elif self.learned_module == 'syntax':
            dataset = [x.pt for x in self.buffer if len(x.pt.sentence) > 1]
            n_iters = int(100)
            print("Learn syntax with %d samples for %d iterations, "%(len(dataset), n_iters), end='', flush=True)
            st = time()
            self.syntax.learn(dataset, n_iters=n_iters)
            print("take %d sec."%(time()-st))

        elif self.learned_module == 'semantics':
            dataset = [[] for _ in range(len(self.semantics.semantics))]
            for ast in self.buffer:
                for node in ast.nodes:
                    xs = [x.res() for x in node.children if x.res() != EMPTY_VALUE]
                    y = node.res()
                    dataset[node.symbol].append((xs, y))
            self.semantics.learn(dataset)

        self.clear_buffer()

if __name__ == '__main__':
    # from utils import SEMANTICS
    # sentences = ['5!-7-4', '1+5!*8', '8*9!+5+1/9/3!*9*5']
    # head = [[1, 2, 4, 2, -1, 4], [1, -1, 3, 4, 1, 4], [1, 4, 3, 1, 6, 4, -1, 8, 10, 8, 13, 12, 10, 15, 13, 6, 15]]
    # for s, dep in zip(sentences, head):
    #     et = AST(s, dep, SEMANTICS)
    #     print(et.res())

    model = Jointer(None)
    from dataset import HINT, HINT_collate
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = HINT('train')
    data_loader = DataLoader(dataset, batch_size=32,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    model.train()
    for sample in tqdm(data_loader):
        # sample = next(iter(val_loader))
        res = model.deduce(sample['img_seq'], sample['len'])
        model.abduce(sample['res'], sample['img_paths'])
        # print(len([1 for x, y in zip(res, sample['res']) if x is not None and x == y]), len(model.buffer))
        # model.clear_buffer()
        model.learn()
    print(len(model.buffer))
