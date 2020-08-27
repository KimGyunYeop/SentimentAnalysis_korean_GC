import argparse
import json
import logging
import os
import glob

import numpy as np
import torch


"""from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    MODEL_ORIGINER,
    init_logger,
    set_seed,
    compute_metrics
)
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from processor import seq_cls_processors as processors
from processor import seq_cls_output_modes as output_modes

from datasets import BaseDataset,KNUDataset, CharBaseDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

tokenizer = TOKENIZER_CLASSES["kocharelectra-base"].from_pretrained(
    "monologg/kocharelectra-base-discriminator"
)

c = argparse.ArgumentParser()
args = c.parse_args()
args.data_dir = "data"
args.task = "nsmc"
args.train_file = "ratings_train.txt"
args.max_seq_len = 50

dataset = BaseDataset(args, tokenizer=tokenizer, mode="train_small")
train_sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
for i, batch in enumerate(dataloader):
    print(batch[1])

print("\n")
for i, batch in enumerate(dataloader):
    print(batch[1])
"""

'''
from konlpy.tag import Okt
txt = "공부를 하면할수록 모르는게 많다는 것을 알게 됩니다."
twitter = Okt()

df = pd.read_csv(os.path.join(args.data_dir,args.task,args.train_file),sep="\t")
df['review'].astype('str')
for line in df["review"]:
    print(twitter.morphs(str(line)))
#[c1, c2, [c3,c4,c5]]

batch_size, seq_len, w2v_dim = 32, 1, 768
data = torch.randn(batch_size, seq_len, w2v_dim)
x1 = data.squeeze()
x1 = x1.repeat(1,batch_size)
x1 = x1.view(batch_size,batch_size,w2v_dim)
print(x1)
print(x1.shape)
x2 = data.squeeze()
x2 = x2.unsqueeze(0)
x2 = x2.repeat(batch_size,1,1)
print(x2)
print(x2.shape)
label = torch.randint(0,2,(batch_size,))
print(label)
#y = torch.randint(0,2,(batch_size, batch_size)) *2 -1
y = label.unsqueeze(0).repeat(batch_size,1)
for i,t in enumerate(y):
    y[i] = (t==t[i]).double() * 2 - 1
loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-1)
loss = loss_fn(x1.view(-1, w2v_dim),
               x2.view(-1, w2v_dim),
               y.view(-1))

print(loss)
print(loss.shape)

batch_size, labels = 32, 2
x1 = torch.randn(batch_size, labels)
x2 = torch.randn(batch_size, labels)
y = torch.ones(batch_size, dtype = torch.int64)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(x1, y)

print(loss)
print(loss.shape)'''

from konlpy.tag import Okt,Kkma

okt = Okt()
print(okt.morphs("우아하여"))
print(okt.morphs("진짜 너무 우아하여 감동입니다"))