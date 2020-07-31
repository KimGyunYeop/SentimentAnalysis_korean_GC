import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle

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
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = self.dataset.at[idx,"review"]
        data = self.tokenizer(str(txt), pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return input_ids, token_type_ids, attention_mask, label, txt

class KOSACDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(KOSACDataset,self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if mode == "train":
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif mode == "dev":
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif mode == "test":
            data_path = os.path.join(args.data_dir, args.task, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        self.polarities, self.intensities = self.get_sentiment_data(self.dataset)

    def convert_sentiment_to_ids(self, mode, all_labels):
        pol2idx = ['None', 'POS', 'NEUT', 'COMP', 'NEG']
        int2idx = ['Medium', 'Low', 'None', 'High']
        all_ids = []
        for labels in all_labels:
            ids = []
            if mode == 'polarity':
                for label in labels:
                    ids.append(pol2idx.index(label))
            elif mode == 'intensity':
                for label in labels:
                    ids.append(int2idx.index(label))

            all_ids.append(ids)

        return all_ids

    def get_sentiment_data(self, dataset):
        tkn2pol = pickle.load(open(os.path.join(self.args.data_dir, self.args.task,'sentiment_data','kosac_polarity.pkl'), 'rb'))
        print(tkn2pol)
        tkn2int = pickle.load(open(os.path.join(self.args.data_dir, self.args.task,'sentiment_data','kosac_intensity.pkl'), 'rb'))
        print(tkn2int)
        polarities = []
        intensities = []

        for i in range(len(dataset)):
            tokens = self.tokenizer._tokenize(str(dataset.at[i,'review']))
            polarity = []
            intensity = []
            polarity.append('None')
            intensity.append('None')
            for token in tokens[:self.maxlen - 2]:
                if token[:2] == '##':
                    tkn = token[2:]
                else:
                    tkn = token
                try:
                    polarity.append(tkn2pol[tkn])
                    intensity.append(tkn2int[tkn])
                except KeyError:
                    polarity.append("None")
                    intensity.append("None")

            polarity.append('None')
            intensity.append('None')

            if self.maxlen - len(polarity) <= 0:
                count = 0
            else:
                count = self.maxlen - len(polarity)

            polarity = polarity + ['None' for i in range(count)]
            intensity = intensity + ['None' for i in range(count)]

            polarities.append(polarity)
            intensities.append(intensity)

        return self.convert_sentiment_to_ids('polarity', polarities), self.convert_sentiment_to_ids('intensity', intensities)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = self.dataset.at[idx,"review"]
        data = self.tokenizer(str(txt), pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        polarity_ids = torch.LongTensor(self.polarities[idx])
        intensity_ids = torch.LongTensor(self.intensities[idx])
        label = self.dataset.at[idx,"rating"]

        return input_ids, token_type_ids, attention_mask, label, polarity_ids, intensity_ids,txt

DATASET_LIST = {
    "LSTM": BaseDataset,
    "LSTM_ATT": BaseDataset,
    "LSTM_ATT_v2": BaseDataset,
    "LSTM_ATT_DOT": BaseDataset,
    "KOSAC_LSTM": KOSACDataset
}
