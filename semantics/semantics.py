import sys
sys.setrecursionlimit(int(1e4))
sys.path.insert(0, "./semantics/dreamcoder")

import random
from collections import defaultdict, Counter 
import json
import math
import os
import datetime
import numpy as np
import torch

from dreamcoder.dreamcoder import commandlineArguments, explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit, numberOfCPUs
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.program import Program, Invented, Primitive
from dreamcoder.frontier import Frontier, FrontierEntry

from dreamcoder.domains.hint.hintPrimitives import McCarthyPrimitives
from dreamcoder.domains.hint import hintPrimitives
from dreamcoder.domains.hint.main import main, list_options, LearnedFeatureExtractor

from utils import SYMBOLS, EMPTY_VALUE, MISSING_VALUE, SYM2PROG

class ProgramWrapper(object):
    def __init__(self, prog):
        try:
            self.fn = prog.evaluate([])
        except RecursionError as e:
            self.fn = None
        self.prog = prog
        self.arity = len(prog.infer().functionArguments())
        self._name = None
        self.cache = {} # used for fast computation
    
    def __call__(self, *inputs):
        if len(inputs) != self.arity or MISSING_VALUE in inputs:
            return MISSING_VALUE
        if inputs in self.cache:
            return self.cache[inputs]
        try:
            fn = self.fn
            for x in inputs:
                fn = fn(x)
        except RecursionError as e:
            print(repr(e))
            fn = MISSING_VALUE
        self.cache[inputs] = fn
        return fn

    def __eq__(self, prog): # only used for removing equivalent semantics
        if self.arity != prog.arity:
            return False
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        # if self.y is not None and prog.y is not None:
        #     assert len(self.y) == len(prog.y) # the program should be evaluated on same examples
        #     return np.mean(self.y[self.y!=None] == prog.y[self.y!=None]) > 0.95
        # return self.prog == prog.prog
        return False

    def __str__(self):
        return "%s %s"%(self.name, self.prog)

    @property
    def name(self):
        if self._name is not None: return self._name
        if isinstance(self.fn, int):
            self._name = str(self.fn)
        else:
            self._name = "fn"
            pass # TODO: assign name based on the function
        return self._name

    def evaluate(self, examples, store_y=True): 
        ys = []
        for exp in examples:
            try:
                y = self(*exp)
            except (TypeError, RecursionError) as e:
                print(repr(e))
                y = MISSING_VALUE
            ys.append(y)
        return ys

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

class NULLProgram(object):
    def __init__(self):
        self.arity = 0
        self.fn = lambda: EMPTY_VALUE
        self.cache = {}

    def __call__(self, *inputs):
        if len(inputs) != 0:
            return MISSING_VALUE
        return self.fn()
    
    def evaluate(self, examples, **kwargs): 
        ys = [self(*e) for e in examples]
        return ys

    def __str__(self):
        return "NULL"

    def solve(self, *args):
        return []

def compute_likelihood(program, examples=None):
    if examples is None:
        return 0., None
    else:
        pred = program.evaluate([e[0] for e in examples], store_y=False)
        gt = np.array([e[1] for e in examples])
        res = pred == gt
        return np.mean(res), np.array(res)

class Semantics(object):
    def __init__(self, idx, program=None, fewshot=False, learnable=True):
        self.idx = idx
        self.examples = []
        self.program = program or NULLProgram()
        self.arity = 2 if self.idx >= 10 and self.idx < 14 else 0
        self.solved = False
        self.fewshot = fewshot
        self.learnable = False if self.idx in [14, 15] else learnable # do not learn parentheses
        self.likelihood = 0. if self.learnable else 1.
        self.cache = {}

    def update_examples(self, examples):
        if len(examples) < 10 and not self.fewshot:
            self.clear()
            return

        examples = [x[:2] for x in examples if len(x[0]) == self.arity] 
        for x, ys in self.cache.items():
            for y in ys:
                ys[y] *= 0.5
        
        for x, y in examples:
            if x not in self.cache:
                self.cache[x] = {}
            if y not in self.cache[x]:
                self.cache[x][y] = 0.
            self.cache[x][y] += 1

        
        conf_examples = []
        conf_thres = 5
        for x, ys in self.cache.items():
            y, conf = max(ys.items(), key=lambda k: k[1])
            if conf >= conf_thres and (conf / sum(ys.values())) > 0.5: # the most confident y occupies 80% of all possible prediction
                conf_examples.append((x, y))
        self.examples = conf_examples

        self.likelihood, self.res = compute_likelihood(self.program, conf_examples)
        self.check_solved()

        gt_program = SYM2PROG[SYMBOLS[self.idx]]
        acc = compute_likelihood(gt_program, examples)[0]
        acc_conf = compute_likelihood(gt_program, conf_examples)[0]
        print(f"Symbol-{self.idx:02d}: arity: {self.arity}, examples (conf): {len(examples)} ({len(conf_examples)}), accuracy (conf): {acc*100:.2f} ({acc_conf*100:.2f})")

    def update_program(self, entry):
        program = ProgramWrapper(entry.program)
        likelihood = compute_likelihood(program, self.examples)[0]
        if (likelihood > self.likelihood) or \
            (likelihood == self.likelihood and len(str(program)) < len(str(self.program))):
            self.program = program
            self.likelihood = likelihood
            self.check_solved()
        # self.program.cache.update({xs:y for xs, y in self.examples})
    
    def check_solved(self):
        if self.program is None:
            self.solved = False
        elif self.likelihood >= 0.95:
            self.solved = True
        elif self.fewshot and self.likelihood >= 0.95 and len(set(self.examples)) >= 10:
            self.solved = True
        else:
            self.solved = False

    def __call__(self, inputs):
        if self.idx in [14, 15]: 
            return EMPTY_VALUE if len(inputs) == 0 else MISSING_VALUE # parentheses

        inputs = [x for x in inputs if x != EMPTY_VALUE]
        inputs = tuple(inputs)
        if self.likelihood > 0.8:
            if isinstance(self.program, NULLProgram) and len(inputs) > 0:
                return MISSING_VALUE
            inputs = [x for x in inputs if x != EMPTY_VALUE]
            return self.program(*inputs)
        elif inputs in self.cache:
            ys = self.cache[inputs]
            candidates = [y[0] for y in ys.items()]
            p = [y[1] for y in ys.items()]
            p = np.array(p) / sum(p)
            output = int(np.random.choice(candidates, p=p))
            return output

        return MISSING_VALUE

    def make_task(self):
        min_examples = 10 if self.arity is not None and self.arity > 0 else 0
        min_examples = min_examples if not self.fewshot else 0
        max_examples = 100
        examples = self.examples
        if len(examples) < min_examples or self.solved or not self.learnable:
            return None
        task_type = arrow(*([tint]*(self.arity + 1)))
        if len(examples) > max_examples:
            wrong_examples = [e for e, r in zip(examples, self.res) if not r]
            right_examples = [e for e, r in zip(examples, self.res) if r]
            right_examples = random.choices(right_examples, k=max_examples-len(wrong_examples))
            examples = wrong_examples + right_examples
            examples = random.sample(examples, k=max_examples)
        return Task(str(self.idx), task_type, examples)

    def solve(self, i, inputs, output_list):
        if len(inputs) != self.arity:
            return []

        def equal(a, b, pos):
            if len(a) != len(b):
                return False
            for j in range(len(a)):
                if j == pos:
                    continue
                if a[j] != b[j]:
                    return False
            return True

        candidates = []
        for xs, y in self.cache.items():
            y = sorted(y.items(), key = lambda p: p[1], reverse=True)[0][0]
            if y in output_list and equal(xs, inputs, i):
                candidates.append(xs[i])
        return candidates

    def clear(self):
        self.examples = []
        self.program = NULLProgram()
        self.solved = False
        self.likelihood = 0.
    
    def save(self):
        model = {'idx': self.idx, 'solved': self.solved, 'likelihood': self.likelihood, 'arity': self.arity}
        model['program'] = None if isinstance(self.program, NULLProgram) else self.program.prog
        return model

    def load(self, model):
        self.idx = model['idx']
        self.solved = model['solved']
        self.likelihood = model['likelihood']
        self.arity = model['arity']
        self.program = NULLProgram() if model['program'] is None else ProgramWrapper(model['program'])

class DreamCoder(object):
    def __init__(self, config=None):
        args = commandlineArguments(
            enumerationTimeout=200, activation='tanh', iterations=1, recognitionTimeout=3600,
            a=3, maximumFrontier=5, topK=2, pseudoCounts=30.0,
            helmholtzRatio=0.5, structurePenalty=1.,
            CPUs=min(numberOfCPUs(), 8),
            extras=list_options)

        args['noConsolidation'] = True
        args.pop("random_seed")
        args['contextual'] = True
        args['biasOptimal'] = True
        args['auxiliaryLoss'] = True
        args['activation'] = "relu"
        args['useDSL'] = False


        extractor = {
            "learned": LearnedFeatureExtractor,
        }[args.pop("extractor")]
        extractor.H = args.pop("hidden")

        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "tmp/%s"%timestamp
        os.system("mkdir -p %s"%outputDirectory)
        
        args.update({
            "featureExtractor": extractor,
            "outputPrefix": "%s/hint"%outputDirectory,
            "evaluationTimeout": 0.0005,
        })
        args.pop("maxTasks")
        args.pop("split")
        
        self.primitives = McCarthyPrimitives()
        if not config.Y_combinator:
            self.primitives = [x for x in self.primitives if 'fix' not in x.name]
        baseGrammar = Grammar.uniform(self.primitives)
        self.grammar = baseGrammar
        self.train_args = args
        self.semantics = [Semantics(i) for i in range(len(SYMBOLS))]
        self.allFrontiers = None
        self.helmholtzFrontiers = None
        self.learn_count = 0

    def __call__(self):
        return self.semantics

    def save(self):
        model = [smt.save() for smt in self.semantics]
        return model

    def load(self, model):
        if model is None:
            return
        assert len(self.semantics) == len(model)
        for i in range(len(self.semantics)):
            self.semantics[i].load(model[i])
    
    def extend(self, n):
        for smt in self.semantics:
            smt.learnable = False
        idx = len(SYMBOLS) - 1
        self.semantics.append(Semantics(idx, fewshot=True))
        self.primitives.extend([Invented(smt.program.prog) for smt in self.semantics if not smt.learnable and smt.arity > 0])

    def rescore_frontiers(self, tasks):
        if self.allFrontiers is None:
            return
        print('Rescoring %d frontiers...'%len(self.allFrontiers))
        id2task = {t.name: t for t in tasks}
        id2frontier = {f.task.name: f for f in self.allFrontiers}
        allFrontiers = {}
        for name in id2task.keys():
            task = id2task[name]
            examples = task.examples
            if name not in id2frontier:
                frontier = Frontier([], task=task)
            else:
                frontier = id2frontier[name]
                frontier.task = task
                for entry in frontier.entries:
                    program = ProgramWrapper(entry.program)
                    entry.logLikelihood = float(np.log(compute_likelihood(program=program, examples=examples)[0]))
                    entry.logPosterior = entry.logLikelihood + entry.logPrior
                frontier.removeLowLikelihood(low=0.1)

            allFrontiers[task] = frontier
        self.allFrontiers = allFrontiers

    def learn(self, dataset):
        self.learn_count += 1
        learning_interval = 10
        tasks = []
        max_arity = 0
        for smt, exps in zip(self.semantics, dataset):
            if not smt.learnable:
                continue
            smt.update_examples(exps)
            if self.learn_count % learning_interval == 0:
                t = smt.make_task()
                if t is not None:
                    tasks.append(t)
                    max_arity = max(smt.arity, max_arity)
        if not tasks: 
            return
        self.train_args['enumerationTimeout'] = 5 if max_arity == 0 else 300
        # self.train_args['iterations'] = 1 if max_arity == 0 else 3
        n_solved = len(['' for t in self.semantics if t.solved or not t.learnable])
        print("Semantics: %d/%d/%d (total/solved/learn)."%(len(self.semantics), n_solved, len(tasks)))
        if len(tasks) == 0:
            self._print_semantics()
            return 
        self._print_tasks(tasks)
        self.update_grammar()
        print(self.grammar)
        # print(self.allFrontiers)
        self.rescore_frontiers(tasks)
        # if self.allFrontiers is not None:
        #     print(self.allFrontiers.values())

        if self.helmholtzFrontiers is not None:
            requests_old ={x.task.request for x in self.helmholtzFrontiers()}
            requests = {t.request for t in tasks}
            # if new requests, discard old helmholtz frontiers
            if requests != requests_old:
                self.helmholtzFrontiers = None

        result = explorationCompression(self.grammar, tasks, allFrontiers=self.allFrontiers, helmholtzFrontiers=self.helmholtzFrontiers, **self.train_args)
        self.allFrontiers = list(result.allFrontiers.values())
        self.helmholtzFrontiers = result.helmholtzFrontiers

        for frontier in result.taskSolutions.values():
            if not frontier.entries: continue
            symbol_idx = int(frontier.task.name)
            # print(frontier)
            self.semantics[symbol_idx].update_program(frontier.bestPosterior)
        # examples = [xs for t in tasks for xs, y in t.examples]
        # self._removeEquivalentSemantics(examples)
        self._removeEquivalentSemantics()
        self._print_semantics()
        # self.grammar = result.grammars[-1]

    def update_grammar(self):
        new_primitives = []
        for smt in self.semantics:
            if '#' in str(smt.program) or '+' in str(smt.program) or '-' in str(smt.program):
                # if '#+-' in the program, the program uses a invented primitive, it is very likely to have a high computation cost.
                # Therefore we don't add this program into primitives, since it might slow the enumeration a lot.
                # it might be resolved by increasing the enumeration time
                continue
            if smt.learnable and smt.solved and smt.arity == 2:
                add = ProgramWrapper(hintPrimitives.add)
                minus0 = ProgramWrapper(hintPrimitives.minus0)
                if np.all([add(*x) == smt.program(*x) for x, y in smt.examples]):
                    new_primitives.append(hintPrimitives.add)
                elif np.all([minus0(*x) == smt.program(*x) for x, y in smt.examples]):
                    new_primitives.append(hintPrimitives.minus0)
                else:
                    new_primitives.append(Invented(smt.program.prog))

        new_grammar = Grammar.uniform(self.primitives + new_primitives)
        if new_grammar != self.grammar:
            self.grammar = new_grammar
            self.helmholtzFrontiers = None
            self.allFrontiers = None
            print("Update grammar with invented programs and set frontiers to none.")
        

    def _print_semantics(self):
        for smt in sorted(self.semantics, key=lambda x: x.idx):
            print("Symbol-%02d: %s %.2f"%(smt.idx, smt.program, smt.likelihood))
            # print("Solved!" if smt.solved else "")

    def _print_tasks(self, tasks):
        for task in tasks:
            print("Symbol-%02d (%s), Samples: %3d, "%(int(task.name), task.request, len(task.examples)), Counter(task.examples).most_common(10))

        # json.dump([t.examples for t in tasks], open('outputs/tasks.json', 'w'))

    def _removeEquivalentSemantics(self, examples=None):
        if examples is not None:
            examples = list(set(examples))
            for smt in self.semantics:
                if smt.program is not None:
                    smt.program.evaluate(examples)
        
        for i in range(len(self.semantics) - 1):
            smt_i = self.semantics[i]
            if smt_i.program is None:
                continue
            for j in range(i+1, len(self.semantics)):
                smt_j = self.semantics[j]
                if smt_j.program is None:
                    continue
                if smt_i.program == smt_j.program:
                    if len(smt_i.examples) >= len(smt_j.examples):
                        smt_j.clear()
                    else:
                        smt_i.clear()
                        break
