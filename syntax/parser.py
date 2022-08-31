# -*- coding: utf-8 -*-

import time
import os
import logging
from collections import Counter

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import math
import random

NULL = '<NULL>'
try:
    from utils import SYMBOLS
    from .general_utils import minibatches
    TOKENS = SYMBOLS + [NULL]
except ImportError:
    from general_utils import minibatches
    TOKENS = list('0123456789+-*/!') + [NULL]

TRANSITIONS = ['L', 'R', 'S'] # Left-Arc, Right-Arc, Shift 
TRAN2ID = {t: i for (i, t) in enumerate(TRANSITIONS)}
ID2TRAN = {i: t for (i, t) in enumerate(TRANSITIONS)}

class ParserModel(nn.Module):
    def __init__(self, n_tokens, embed_size=50, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.
        @param n_tokens (int): the size of the vocabulary
        @param n_features (int): number of input features
        @param embed_size (int): number of embedding units
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_tokens = n_tokens
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(n_tokens, embed_size)
        self.model = nn.Sequential(
            nn.Linear(embed_size * n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, n_classes),
        )

    def extend(self, n):
        old_embeddings = self.embeddings
        embeddings = nn.Embedding(self.n_tokens+n, self.embed_size)
        embeddings.weight.data[:self.n_tokens-1] = old_embeddings.weight.data[:self.n_tokens-1]
        embeddings.weight.data[-1] = old_embeddings.weight.data[-1] # the last token is always NULL for padding
        self.embeddings = embeddings
        self.n_tokens += n

    def forward(self, t):
        """ Run the model forward.
        @param t (Tensor): input tensor of tokens (batch_size, n_features)
        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        x = self.embeddings(t)
        logits = self.model(x.view(x.shape[0], -1))
        return logits

class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""
    def __init__(self):

        self.n_trans = len(TRANSITIONS)
        self.n_features = 18
        self.n_tokens = len(TOKENS)
        self.tok2id = {v: k for (k, v) in enumerate(TOKENS)}

        self.model = ParserModel(n_tokens=self.n_tokens, n_features=self.n_features)
        self.device = torch.device('cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def save(self, save_optimizer=True):
        saved = {'model': self.model.state_dict()}
        if save_optimizer:
            saved['optimizer'] = self.optimizer.state_dict()
        return saved
    
    def load(self, loaded):
        self.model.load_state_dict(loaded['model'])
        if 'optimizer' in loaded:
            self.optimizer.load_state_dict(loaded['optimizer'])

    def extend(self, n):
        self.model.extend(n)
        TOKENS = SYMBOLS + [NULL]
        self.n_tokens = len(TOKENS)
        self.tok2id = {v: k for (k, v) in enumerate(TOKENS)}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, amsgrad=True)

    def __call__(self, sentences):
        return self.parse(sentences)

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.tok2id[w] for w in ex['expr']]
            head = ex['head']
            vec_examples.append({'word': word, 'head': head})
        return vec_examples

    def extract_features(self, stack, buf, arcs, sent):
        """ extract features for current state, used by neural network to predict next action,
            check Section 3.1 in https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf for more details.
        """
        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        null_idx = self.tok2id[NULL]
        features = [null_idx] * (3 - len(stack)) + [sent[x] for x in stack[-3:]]
        features += [sent[x] for x in buf[:3]] + [null_idx] * (3 - len(buf))
        for i in range(2):
            if i < len(stack):
                k = stack[-i-1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(sent[lc[0]] if len(lc) > 0 else null_idx)
                features.append(sent[rc[0]] if len(rc) > 0 else null_idx)
                features.append(sent[lc[1]] if len(lc) > 1 else null_idx)
                features.append(sent[rc[1]] if len(rc) > 1 else null_idx)
                features.append(sent[llc[0]] if len(llc) > 0 else null_idx)
                features.append(sent[rrc[0]] if len(rrc) > 0 else null_idx)
            else:
                features += [null_idx] * 6

        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, head):
        if len(stack) < 2:
            return self.n_trans - 1 if len(buf) > 0 else None

        i0 = stack[-1]
        i1 = stack[-2]
        h0 = head[i0]
        h1 = head[i1]

        if h1 == i0: # left-arc
            tran = 'L'
        elif h0 == i1 and (not any([x for x in buf if head[x] == i0])): # right-arc
            tran = 'R'
        else:
            tran = 'S' if len(buf) > 0 else None  # shift
        return TRAN2ID[tran] if tran else None

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word'])

            # arcs = {(h, t, label)}
            stack = []
            buf = [i for i in range(n_words)]
            arcs = []
            instances = []
            for i in range(n_words * 2 - 1):
                gold_t = self.get_oracle(stack, buf, ex['head'])
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex['word']),
                                  legal_labels, gold_t))
                if gold_t == self.n_trans - 1:
                    stack.append(buf.pop(0))
                elif gold_t == 0:
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack.pop(-2)
                else:
                    arcs.append((stack[-2], stack[-1], gold_t))
                    stack.pop(-1)

            assert len(stack) == 1 and len(buf) == 0
            if len(stack) == 1 and len(buf) == 0:
                succ += 1
                all_instances += instances

        return all_instances

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) >= 2 else [0]) # left-arc
        labels += ([1] if len(stack) >= 2 else [0]) # right-arc
        labels += [1] if len(buf) > 0 else [0] # shift
        return labels

    def parse(self, sentences, batch_size=5000):
        parses = []
        partial_parses = [PartialParse(sen) for sen in sentences]
        unfinished_parses = partial_parses[:]
        while unfinished_parses:
            minibatch_parses = unfinished_parses[:batch_size]
            unfinished_parses = unfinished_parses[batch_size:]
            parse_index = list(range(len(minibatch_parses)))
            batch_parses = [None] * len(minibatch_parses)
            while minibatch_parses:
                transitions, probs = self.predict(minibatch_parses)
                probs = probs.detach().cpu().numpy()
                transitions = transitions.detach().cpu().numpy()
                index_rm = []
                for i in range(len(minibatch_parses)):
                    minibatch_parses[i].parse_step(transitions[i], probs[i])
                    if minibatch_parses[i].finish:
                        batch_parses[parse_index[i]] = minibatch_parses[i]
                        index_rm.append(i)
                for index in sorted(index_rm, reverse=True):   #we need to use 'del' in reverse order 
                    del minibatch_parses[index]
                    del parse_index[index]
            parses.extend(batch_parses)
        return parses

    def predict(self, partial_parses):
        mb_x = [self.extract_features(p.stack, p.buffer, p.dependencies, p.sentence) for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_x = torch.from_numpy(mb_x).long().to(self.device)
        mb_l = [self.legal_labels(p.stack, p.buffer) for p in partial_parses]

        logits = self.model(mb_x)
        probs = nn.functional.softmax(logits, dim=-1)
        probs *= torch.tensor(mb_l, dtype=probs.dtype, device=probs.device)
        probs /= probs.sum(-1, keepdim=True)

        if self.model.training:
            m = Categorical(probs=probs)
            preds = m.sample()
        else:
            preds = torch.argmax(probs, -1)

        return preds, probs

    def evaluate(self, dataset):
        sentences = [x['word'] for x in dataset]
        parses = self.parse(sentences)

        UAS = all_tokens = 0.0
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = parses[i].dependencies
                for pred_h, gold_h in zip(head, ex['head']):
                    UAS += 1 if pred_h == gold_h else 0
                    all_tokens += 1
                prog.update(i + 1)
        UAS /= all_tokens
        return UAS

    def learn(self, dataset, n_iters=100):
        train_data = self.create_instances([{'word': x.sentence, 'head': x.head} for x in dataset])

        batch_size = 1024
        n_epochs = int(math.ceil(batch_size * n_iters / len(train_data)))
        n_epochs = max(n_epochs, 5) # run at least 5 epochs
        print(n_epochs, "epochs, ", end='')
        self.model.train() # Places model in "train" mode, i.e. apply dropout layer
        for epoch in range(n_epochs):
            for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
                train_x = torch.from_numpy(train_x).long()
                train_y = torch.from_numpy(train_y.nonzero()[1]).long()
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                output_y = self.model(train_x)
                loss = self.criterion(output_y, train_y)

                self.optimizer.zero_grad()   # remove any baggage in the optimizer
                loss.backward()
                self.optimizer.step()

class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.
        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        self.sentence = sentence
        self.stack = [] 
        self.buffer = list(range(len(sentence)))
        self.dependencies = []
        self.transitions = []
        self.probs = []
        self.finish = False # whether the parse has finished

    def parse_step(self, transition, prob=None):
        """Performs a single parse step by applying the given transition to this partial parse
        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """
        transition_prob = prob[transition]
        if transition == 0: # Left-Arc
            d=(self.stack[-1],self.stack[-2],transition_prob)
            self.dependencies.append(d)
            self.stack.pop(-2)
        elif transition == 1: # Right-Arc
            d=(self.stack[-2],self.stack[-1],transition_prob)
            self.dependencies.append(d)
            self.stack.pop(-1)
        elif transition == 2: # Shift
            self.stack.append(self.buffer.pop(0))
        self.transitions.append(transition)
        self.probs.append(prob)
        if len(self.buffer) == 0 and len(self.stack) == 1:
            self.finish = True
            self.compute_head()
    
    def compute_head(self):
        self.head = [-1] * len(self.sentence)
        for arc in self.dependencies:
            h, t = arc[:2]
            self.head[t] = h

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dsependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies
