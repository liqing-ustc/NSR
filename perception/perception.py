import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset
from tqdm import trange, tqdm
import numpy as np
from . import resnet_scan, lenet_scan


class Perception(object):
    def __init__(self, domain):
        super(Perception, self).__init__()
        self.n_class = len(domain.i2w)
        self.load_fn = domain.load_image
        self.model = SentenceEncoder(self.n_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.device('cpu')
        self.training = False
        self.min_examples = 200
        self.selflabel_dataset = None
    
    def train(self):
        # self.model.train()
        self.model.eval()
        self.training = True

    def eval(self):
        self.model.eval()
        self.training = False
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def save(self, save_optimizer=True):
        saved = {'model': self.model.state_dict()}
        if save_optimizer:
            saved['optimizer'] = self.optimizer.state_dict()
        return saved
    
    def load(self, loaded, image_encoder_only=False):
        if image_encoder_only:
            self.model.image_encoder.load_state_dict(loaded['model'])
        else:
            self.model.load_state_dict(loaded['model'])
        if 'optimizer' in loaded:
            self.optimizer.load_state_dict(loaded['optimizer'])

    def extend(self, n):
        self.n_class += n
        self.model.extend(n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def selflabel(self, dataset):
        print('Self labeling from pretrained perception model...')
        symbols = [(x, y) for sample in dataset for x, y in zip(sample['img_paths'], sample['sentence'])]
        dataloader = torch.utils.data.DataLoader(ImageSet(symbols, self.load_fn), batch_size=512,
                         shuffle=False, drop_last=False, num_workers=8)
        with torch.no_grad():
            self.eval()
            prob_all = []
            for img, _ in tqdm(dataloader):
                img = img.to(self.device)
                prob = self.model.image_encoder(img)
                prob = nn.functional.softmax(prob, dim=-1)
                prob_all.append(prob)
            prob_all = torch.cat(prob_all)
        
        confidence = 0.95
        selflabel_dataset = {}
        probs, preds = torch.max(prob_all, dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()
        for cls_id in range(self.n_class):
            idx_list = np.where(preds == cls_id)[0]
            idx_list = sorted(idx_list, key=lambda x: probs[x], reverse=True)
            idx_list = [i for i in idx_list if probs[i] >= confidence]
            images = [symbols[i][0] for i in idx_list]
            # images = list(set(images))
            labels = [symbols[i][1] for i in idx_list]
            acc = np.mean(np.array(labels) == cls_id)
            selflabel_dataset[cls_id] = [(x, cls_id) for x in images]
            print("Add %d samples for class %d, acc %.2f."%(len(images), cls_id, acc))
        img2cls = {img: cls_id for examples in selflabel_dataset.values() for img, cls_id in examples}
        dataset = [(sample['img_paths'], list(map(lambda x: img2cls.get(x, None), sample['img_paths']))) for sample in dataset]
        dataset = [(x, y) for x, y in dataset if None not in y]
        self.selflabel_dataset = dataset
        print(f'Add {len(dataset)} self-labelled examples for perception.')
        self.learn([], n_epochs=50)

    def __call__(self, images, src_len):
        logits = self.model(images, src_len)
        # probs = torch.sigmoid(logits)
        probs = nn.functional.softmax(logits, dim=-1)
        if self.training:
            m = Categorical(probs=probs)
            preds = m.sample()
        else:
            preds = torch.argmax(probs, -1)

        return preds, probs

    def learn(self, dataset=[], n_epochs=1):
        dataset = dataset + self.selflabel_dataset
        batch_size = 32
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')

        print(n_epochs, "epochs, ", end='')
        dataset = ImageSeqSet(dataset, self.load_fn)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=collate, num_workers=8)
        self.model.train()
        for epoch in range(n_epochs):
            for sample in train_dataloader:
                img_seq = sample['img_seq'].to(self.device)
                sentence = sample['sentence'].to(self.device)
                length = sample['length']
                logit = self.model(img_seq, length)
                # label = nn.functional.one_hot(label, num_classes=self.n_class).type_as(logit)
                loss = criterion(logit, sentence)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

class SymbolNet(nn.Module):
    def __init__(self, n_class):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SentenceEncoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.image_encoder = resnet_scan.make_model(n_class)
        input_dim, emb_dim, hidden_dim, layers, dropout = 512, 128, 128, 2, 0.5
        self.n_token = n_class + 3
        self.embedding = nn.Embedding(self.n_token, emb_dim)
        self.fc_in = nn.Linear(input_dim, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(2 * hidden_dim, n_class)
    
    def forward(self, src, src_len):
        src = self.image_encoder.backbone(src)
        src = self.fc_in(src)

        max_len = max(src_len)
        current = 0
        padded_src = []
        emb_start = self.embedding(torch.tensor([self.n_token - 3]).to(src.device))
        emb_end = self.embedding(torch.tensor([self.n_token - 2]).to(src.device))
        emb_null = self.embedding(torch.tensor([self.n_token - 1]).to(src.device))
        for l in src_len:
            current_input = src[current:current+l]
            current_input = [emb_start, current_input, emb_end] + [emb_null] * (max_len - l) 
            current_input = torch.cat(current_input)
            padded_src.append(current_input)
            current += l
        src = torch.stack(padded_src)

        outputs, _ = self.encoder(src)
        logits = self.fc_out(outputs)
        unroll_logits = [p[1:l+1] for l, p in zip(src_len, logits)] # the first token is START
        logits = torch.cat(unroll_logits)
        return logits


class ImageSet(Dataset):
    def __init__(self, dataset, load_fn):
        super(ImageSet, self).__init__()
        self.dataset = dataset
        self.load_fn = load_fn

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = self.load_fn(img_path)
        return img, label

    def __len__(self):
        return len(self.dataset)

class ImageSeqSet(Dataset):
    def __init__(self, dataset, load_fn):
        super().__init__()
        self.dataset = dataset
        self.load_fn = load_fn

    def __getitem__(self, index):
        sample = self.dataset[index]
        img_paths, labels = sample
        images = []
        for img_path in img_paths:
            img = self.load_fn(img_path)
            images.append(img)

        return {'img_seq': images, 'sentence': labels}

    def __len__(self):
        return len(self.dataset)
    

def collate(batch):
    img_seq_list = []
    sentence_list = []
    length_list = []
    for sample in batch:
        img_seq_list.extend(sample['img_seq'])
        sentence_list.extend(sample['sentence'])
        length_list.append(len(sample['sentence']))

        del sample['img_seq']
        del sample['sentence']
    
    batch = {}
    batch['img_seq'] = torch.stack(img_seq_list)
    batch['sentence'] = torch.tensor(sentence_list)
    batch['length'] = torch.tensor(length_list)
    return batch



