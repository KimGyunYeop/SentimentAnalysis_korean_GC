import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, datapath, tokenizer, maxlen):
        super(BaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.dataset = pd.read_csv(datapath,encoding="utf8", sep="\t")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = self.dataset.at[idx,"review"]
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return input_ids, token_type_ids, attention_mask, label
