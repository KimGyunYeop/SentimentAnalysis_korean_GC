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

from datasets import BaseDataset,KNUDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

tokenizer = TOKENIZER_CLASSES["koelectra-base"].from_pretrained(
    "monologg/koelectra-base-discriminator"
)

print(tokenizer._tokenize("김태우가 멋있게 나와서 10점만점에 10쩜 평점보단 재밌음"))
print(tokenizer("김태우가 멋있게 나와서 10점만점에 10쩜 평점보단 재밌음")["input_ids"])
print(tokenizer.convert_ids_to_tokens(tokenizer("김태우가 멋있게 나와서 10점만점에 10쩜 평점보단 재밌음")["input_ids"]))

c = argparse.ArgumentParser()
args = c.parse_args()
args.data_dir = "data"
args.task = "nsmc"
args.train_file = "ratings_train.txt"
args.max_seq_len = 50

dataset = KNUDataset(args, tokenizer=tokenizer, mode="train_small")
dataloader = DataLoader(dataset, batch_size=1)

for i, batch in enumerate(dataloader):
    print(batch)