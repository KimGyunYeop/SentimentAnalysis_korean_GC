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

from datasets import BaseDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

tokenizer = TOKENIZER_CLASSES["koelectra-base"].from_pretrained(
    "monologg/koelectra-base-discriminator",
    do_lower_case=True,
)

dataset = BaseDataset(datapath="data/nsmc/ratings_train.txt", tokenizer=tokenizer, maxlen=150)
dataloader = DataLoader(dataset, batch_size=2)

for i, (input_ids, token_type_ids, attention_mask,  label) in enumerate(dataloader):
    print(input_ids)
    print(label)