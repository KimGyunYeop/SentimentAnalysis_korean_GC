import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from model import *
import pickle
import pandas as pd

from transformers import (
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

dataset = CharBaseDataset(args, tokenizer=tokenizer, mode="train_small")
dataloader = DataLoader(dataset, batch_size=1)
'''
for i, batch in enumerate(dataloader):
    print(batch[0])
    print(batch[1])
'''

from konlpy.tag import Okt
txt = "공부를 하면할수록 모르는게 많다는 것을 알게 됩니다."
twitter = Okt()

df = pd.read_csv(os.path.join(args.data_dir,args.task,args.train_file),sep="\t")
df['review'].astype('str')
for line in df["review"]:
    print(twitter.morphs(str(line)))
#[c1, c2, [c3,c4,c5]]