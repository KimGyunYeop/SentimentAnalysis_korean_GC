import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(BaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if mode == "train":
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif mode == "dev":
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif mode == "test":
            data_path = os.path.join(args.data_dir, args.task, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")[:1000]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = self.dataset.at[idx,"review"]
        data = self.tokenizer(str(txt), pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return input_ids, token_type_ids, attention_mask, label

DATASET_LIST = {
    "LSTM": BaseDataset,
    "LSTM_ATT": BaseDataset,
    "LSTM_ATT_v2": BaseDataset,
    "LSTM_ATT_DOT": BaseDataset
}
