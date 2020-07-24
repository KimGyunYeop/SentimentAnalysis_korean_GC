import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class baseDataset(Dataset):
    def __init__(self, datapath, tokenizer, maxlen):
        super(baseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.dataset = pd.read_csv(datapath,encoding="utf8", sep="\t")

    def __len__(self):
        print(len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()

        one_data = self.dataset.loc[idx,:]
        data = torch.LongTensor(self.tokenizer.batch_encode_plus(one_data[0], maxlen=self.maxlen,pad_to_max_length=True))
        print(data)
        label = torch.LongTensor(one_data[1])

        return (data,)+(label)
