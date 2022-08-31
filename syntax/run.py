#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
import json
import numpy as np

from torch import nn, optim
import torch
from tqdm import tqdm

from parser import Parser
from general_utils import minibatches


class Config(object):
    data_path = '/home/qing/Desktop/Closed-Loop-Learning/CLL-NeSy/data'
    train_file = 'expr_train.json'
    dev_file = 'expr_val.json'
    test_file = 'expr_test.json'

def load_and_preprocess_data(reduced=True):
    config = Config()

    print("Loading data...",)
    start = time.time()
    train_set = json.load(open(os.path.join(config.data_path, config.train_file)))
    dev_set = json.load(open(os.path.join(config.data_path, config.dev_file)))
    test_set = json.load(open(os.path.join(config.data_path, config.test_file)))
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    parser = Parser()

    print("Vectorizing data...",)
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...",)
    start = time.time()
    train_examples = parser.create_instances(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    return parser, train_examples, dev_set, test_set,


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0
    optimizer = torch.optim.Adam(parser.model.parameters(), lr=lr, amsgrad=True)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.
    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train() # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()
            output_y = parser.model(train_x)
            loss = loss_func(output_y, train_y)
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS = parser.evaluate(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    #debug = True
    debug = False

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    train(parser, train_data, dev_data, batch_size=1024, n_epochs=1, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS = parser.evaluate(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
