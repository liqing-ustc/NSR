import time
from tqdm import tqdm, trange
from collections import Counter, OrderedDict

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from datasets import get_dataset, MISSING_VALUE
from jointer import Jointer

import torch
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser('Neural-Symbolic Recursive Machine')
    parser.add_argument('--wandb', type=str, default='NSR', help='the project name for wandb.')
    parser.add_argument('--dataset', default='hint', choices=['scan', 'pcfg', 'hint'], help='the dataset name.')
    parser.add_argument('--resume', type=str, default=None, help='Resumes training from checkpoint.')
    parser.add_argument('--perception_pretrain', type=str, help='initialize the perception from pretrained models.',
                        default='perception/pretrained_model/model_78.2.pth.tar')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory for storing checkpoints')
    parser.add_argument('--save_model', default='1', choices=['0', '1'])
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")

    parser.add_argument('--train_size', type=float, default=None, help="what perceptage of train data is used.")
    parser.add_argument('--max_op_train', type=int, default=None, help="The maximum number of ops in train.")
    parser.add_argument('--main_dataset_ratio', type=float, default=0, 
            help="The percentage of data from the main training set to avoid forgetting in few-shot learning.")
    parser.add_argument('--fewshot', default=None, choices=list('xyabcd'), help='fewshot concept.')

    parser.add_argument('--perception', default='0', choices=['0', '1'], help='whether to provide perfect perception, i.e., no need to learn')
    parser.add_argument('--syntax', default='0', choices=['0', '1'], help='whether to provide perfect syntax, i.e., no need to learn')
    parser.add_argument('--semantics', default='0', choices=['0', '1'], help='whether to provide perfect semantics, i.e., no need to learn')
    parser.add_argument('--curriculum', default='1', choices=['0', '1'], help='whether to use the pre-defined curriculum')
    parser.add_argument('--Y_combinator', default='1', choices=['0', '1'], help='whether to use the recursion primitive (Y-combinator) in dreamcoder')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--epochs_eval', type=int, default=10, help='how many epochs per evaluation')

    args = parser.parse_args()
    args.wandb = args.wandb + '-' + args.dataset
    args.save_model = args.save_model == '1'
    args.curriculum = args.curriculum == '1'
    args.perception = args.perception == '1'
    args.syntax = args.syntax == '1'
    args.semantics = args.semantics == '1'
    args.Y_combinator = args.Y_combinator == '1'
    return args

from nltk.tree import Tree
def draw_parse(sentence, head):
    def build_tree(pos):
        children = [i for i, h in enumerate(head) if h == pos]
        return Tree(sentence[pos], [build_tree(x) for x in children])
    
    root = head.index(-1)
    tree = build_tree(root)
    return tree

def evaluate(model, dataloader, n_steps=1, log_prefix='val'):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    sent_all = []
    sent_pred_all = []

    dep_all = []
    dep_pred_all = []

    metrics = OrderedDict()

    with torch.no_grad():
        for sample in tqdm(dataloader):
            res = sample['res']
            sent = sample['sentence']
            dep = sample['head']

            res_preds, sent_preds, dep_preds = model.deduce(sample, n_steps=n_steps)
            
            res_pred_all.extend(res_preds)
            res_all.extend(res)
            sent_pred_all.extend(sent_preds)
            sent_all.extend(sent)
            dep_pred_all.extend(dep_preds)
            dep_all.extend(dep)

    pred = res_pred_all
    gt = res_all
    result_acc = np.mean([x == y  for x, y in zip(pred, gt)])
    print("Percentage of missing result: %.2f"%(np.mean([x is MISSING_VALUE for x in pred]) * 100))
    
    pred = [y for x in sent_pred_all for y in x]
    gt = [y for x in sent_all for y in x]
    perception_acc = np.mean([x == y for x,y in zip(pred, gt)])

    pred = [y for x in dep_pred_all for y in x]
    gt = [y for x in dep_all for y in x]
    head_acc = np.mean(np.array(pred) == np.array(gt))

    metrics['result_acc/avg'] = result_acc
    metrics['perception_acc/avg'] = perception_acc
    metrics['head_acc/avg'] = head_acc
    wandb.log({f'{log_prefix}/{k}': v for k, v in metrics.items()})
    
    print("error cases:")
    errors = [i for i in range(len(res_all)) if res_pred_all[i] != res_all[i]]
    if len(errors) == 0:
        errors = [i for i in range(len(res_all)) if dep_pred_all[i] != dep_all[i]]
    for i in errors[:3]:
        expr = ' '.join([model.config.domain.i2w[x] for x in sent_all[i]])
        expr_pred = ' '.join([model.config.domain.i2w[x] for x in sent_pred_all[i]])
        print(expr)
        print(expr_pred)
        print(dep_all[i])
        print(dep_pred_all[i])
        print(res_all[i])
        print(res_pred_all[i])
        print()
        # tree = draw_parse(expr_pred, dep_pred_all[i])
        # tree.draw()


    return perception_acc, head_acc, result_acc

def train(model, args, st_epoch=0):
    best_acc = 0.0
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(args.train_set, batch_size=batch_size,
                         shuffle=True, num_workers=4, collate_fn=args.domain.collate)
    eval_dataloader = torch.utils.data.DataLoader(args.val_set, batch_size=batch_size,
                         shuffle=False, num_workers=4, collate_fn=args.domain.collate)
    
    max_len = float("inf")
    if args.curriculum:
        curriculum_strategy = dict([
            # (0, 7)
            # (0, 3),
            (0, 10),
            (20, 20),
            (40, float('inf')),
        ])
        print("Curriculum:", sorted(curriculum_strategy.items()))
        for e, l in sorted(curriculum_strategy.items(), reverse=True):
            if st_epoch >= e:
                max_len = l
                break
        train_set.filter_by_len(max_len=max_len)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=4, collate_fn=args.domain.collate)
    
    ##########evaluate init model###########
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
    print('Iter {}: {} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format(0, 'val', 100*perception_acc, 100*head_acc, 100*result_acc))
    ########################################

    for epoch in range(st_epoch, args.epochs):
        if args.curriculum and epoch in curriculum_strategy:
            max_len = curriculum_strategy[epoch]
            train_set.filter_by_len(max_len=max_len)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=4, collate_fn=args.domain.collate)
            if len(train_dataloader) == 0:
                continue

        since = time.time()
        print('-' * 30)
        print('Epoch {}/{} (max_len={}, data={})'.format(epoch, args.epochs - 1, max_len, len(train_set)))

        for _ in range(len(model.learning_schedule)):
            with torch.no_grad():
                model.train()
                train_result_acc = []
                train_perception_acc = []
                train_head_acc = []
                n_samples = 0
                for sample in tqdm(train_dataloader):
                    res = sample['res']
                    res_pred, sent_pred, head_pred = model.deduce(sample)
                    model.abduce(res)
                    acc = np.mean([x == y  for x, y in zip(res_pred, res)])
                    train_result_acc.append(acc)

                    sent_pred = [y for x in sent_pred for y in x]
                    sent = [y for x in sample['sentence'] for y in x]
                    acc = np.mean(np.array(sent_pred) == np.array(sent))
                    train_perception_acc.append(acc)

                    head_pred = [y for x in head_pred for y in x]
                    head = [y for x in sample['head'] for y in x]
                    acc = np.mean(np.array(head_pred) == np.array(head))
                    train_head_acc.append(acc)

                    n_samples += len(res)
                    if len(model.buffer) > 1e4:
                        # get enough examples to learn
                        break

                train_result_acc = np.mean(train_result_acc)
                train_perception_acc = np.mean(train_perception_acc)
                train_head_acc = np.mean(train_head_acc)
                abduce_acc = len(model.buffer) / n_samples
            
            wandb.log({'train/result_acc': train_result_acc, 
                       'train/perception_acc': train_perception_acc, 
                       'train/head_acc': train_head_acc, 
                        f'train/abduce_acc/{model.learned_module}': abduce_acc})
            
            model.learn()
            model.epoch += 1
            
        if ((epoch+1) % args.epochs_eval == 0) or (epoch+1 == args.epochs):
            perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
            print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('val', 100*perception_acc, 100*head_acc, 100*result_acc))
            if result_acc > best_acc:
                best_acc = result_acc

            if args.save_model:
                model_path = os.path.join(args.ckpt_dir, "model_%03d.p"%(epoch + 1))
                model.save(model_path, epoch=epoch+1)
                
        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    n_steps = 1
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, n_steps)
    print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('val', 100*perception_acc, 100*head_acc, 100*result_acc))

    # Test
    print('-' * 30)
    print('Evaluate on test set...')
    eval_dataloader = torch.utils.data.DataLoader(args.test_set, batch_size=batch_size,
                         shuffle=False, num_workers=4, collate_fn=args.domain.collate)
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, n_steps, log_prefix='test')
    print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('test', 100*perception_acc, 100*head_acc, 100*result_acc))

    print('Final model:')
    model.print()
    return



if __name__ == "__main__":
    args = parse_args()
    sys.argv = sys.argv[:1]
    wandb.init(project=args.wandb, dir=args.output_dir, config=vars(args))
    ckpt_dir = os.path.join(wandb.run.dir, '../ckpt')
    os.makedirs(ckpt_dir)
    args.ckpt_dir = ckpt_dir
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    domain = get_dataset(args.dataset)
    args.domain = domain
    model = Jointer(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_set = domain('train', n_sample=args.train_size)
    val_set = domain('val')
    test_set = domain('test')
    print('train:', len(train_set), 'val:', len(val_set), 'test:', len(test_set))

    st_epoch = 0
    if args.resume:
        st_epoch = model.load(args.resume)
        if st_epoch is None:
            st_epoch = 0


    model.print()
    wandb.log({'train_examples': len(train_set)})

    args.train_set = train_set
    args.val_set = val_set
    args.test_set = test_set

    train(model, args, st_epoch=st_epoch)

