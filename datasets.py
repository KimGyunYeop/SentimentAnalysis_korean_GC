import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import numpy as np

from konlpy.tag import Twitter

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(BaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        if "small" in mode:
            self.dataset = self.dataset[:10000]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = str(self.dataset.at[idx,"review"])
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return (input_ids, token_type_ids, attention_mask, label),txt

class CharBaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(CharBaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.word_tokenizer = Twitter()
        self.maxlen = 128
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        if "small" in mode:
            self.dataset = self.dataset[:10000]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = str(self.dataset.at[idx,"review"])
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        char_token = self.tokenizer._tokenize(txt)
        word_token = self.word_tokenizer.morphs(txt)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"rating"]

        return (input_ids, token_type_ids, attention_mask, label),[txt, char_token, word_token]

class KOSACDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(KOSACDataset,self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.test_file)

        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        if "small" in mode:
            self.dataset = self.dataset[:10000]
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

    def convert_ids_to_intensity(self, ids):
        int2idx = ['Medium', 'Low', 'None', 'High']
        dict_int2idx = {x:y for x,y in enumerate(int2idx)}
        new_ids = [dict_int2idx[x] for x in ids]

        return new_ids

    def convert_ids_to_polarity(self, ids):
        pol2idx = ['None', 'POS', 'NEUT', 'COMP', 'NEG']
        dict_pol2idx = {x:y for x,y in enumerate(pol2idx)}
        new_ids = [dict_pol2idx[x] for x in ids]

        return new_ids

    def get_sentiment_data(self, dataset):
        tkn2pol = pickle.load(open(os.path.join(self.args.data_dir, self.args.task,'sentiment_data','kosac_polarity.pkl'), 'rb'))
        tkn2int = pickle.load(open(os.path.join(self.args.data_dir, self.args.task,'sentiment_data','kosac_intensity.pkl'), 'rb'))
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
        txt = str(self.dataset.at[idx,"review"])
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        polarity_ids = torch.LongTensor(self.polarities[idx])
        intensity_ids = torch.LongTensor(self.intensities[idx])
        label = self.dataset.at[idx,"rating"]

        return (input_ids, token_type_ids, attention_mask, label, polarity_ids, intensity_ids),txt

class KNUDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(KNUDataset,self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.dev_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.task, args.test_file)

        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")
        if "small" in mode:
            self.dataset = self.dataset[:10000]
        self.polarities = self.get_sentiment_data(self.dataset)

    def find_sub_list(self, sl, l):
        results = []
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                results.append((ind, ind + sll - 1))

        return results

    def get_sentiment_data(self, dataset):
        tkn2pol = pd.read_csv(os.path.join("lexicon","KNU_origin.csv"),header=None,sep="\t")
        tkn2pol_trim = pd.read_csv(os.path.join("lexicon", "KNU_origin.csv"), header=None, sep="\t")
        key2pol = {tuple(self.tokenizer._tokenize(str(word))):pol for word,pol in zip(tkn2pol[0],tkn2pol[1])}
        key2pol_trim = {tuple(self.tokenizer._tokenize(str(word).replace(" ",""))):pol for word,pol in zip(tkn2pol_trim[0],tkn2pol_trim[1])}
        print(len(key2pol))
        key2pol.update(key2pol_trim)
        print(len(key2pol))
        sorted_key = sorted(key2pol.keys() ,key=len)
        for i in sorted_key:
            print(len(i))
        polarities = []
        '''
        for i in range(len(dataset)):
            txt = str(dataset.at[i,'review'])
            tokens = self.tokenizer._tokenize(txt)
            txt = txt.replace(" ", "")
            char_polarity = []
            for j in range(len(txt)):
                if len(char_polarity) > j:
                    continue
                for k in range(7,-1,-1):
                    if k == 0:
                        char_polarity.extend([0])
                    if txt[j:j+k] in key_list:
                        char_polarity.extend(tkn2pol[txt[j:j+k]]*k)
            polarity = ['None']
            count = 0
            for token in tokens[:self.maxlen - 2]:
                if token[:2] == '##':
                    tkn = token[2:]
                else:
                    tkn = token
                if tkn == "[UNK]":
                    polarity.append(0)
                char_pol_list = char_polarity[count:count+len(tkn)]
                pol = max(set(char_pol_list), key=char_pol_list.count)
                polarity.append(pol)
                count = count+len(tkn)'''

        for i in range(len(dataset)):
            txt = str(dataset.at[i,'review'])
            tokens = self.tokenizer._tokenize(txt)
            polarity = [0]*self.maxlen
            for key in sorted_key:
                one_polarity_list = self.find_sub_list(tokens, list(key))
                print(one_polarity_list)
                for start,end in one_polarity_list:
                    print(polarity)
                    print([key2pol[key]]*(end-start))
                    polarity[start:end+1] = [key2pol[key]]*(end-start)
                    print(polarity)


            if self.maxlen - len(polarity) <= 0:
                count = 0
            else:
                count = self.maxlen - len(polarity)

            polarity = polarity + ['None' for i in range(count)]

            polarities.append(polarity)

        return polarities

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = str(self.dataset.at[idx,"review"])
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        token_type_ids = torch.LongTensor(data["token_type_ids"])
        attention_mask = torch.LongTensor(data["attention_mask"])
        polarity_ids = torch.LongTensor(self.polarities[idx])
        label = self.dataset.at[idx,"rating"]

        return (input_ids, token_type_ids, attention_mask, label, polarity_ids),txt

DATASET_LIST = {
    "BASEELECTRA": BaseDataset,
    "BASEELECTRA_COS": BaseDataset,

    "LSTM": BaseDataset,
    "LSTM_ATT": BaseDataset,
    "LSTM_ATT_v2": BaseDataset,
    "LSTM_ATT_DOT": BaseDataset,
    "LSTM_ATT2" : BaseDataset,

    "LSTM_KOSAC": KOSACDataset,
    "LSTM_ATT_KOSAC": KOSACDataset,
    "LSTM_ATT_v2_KOSAC": KOSACDataset,
    "LSTM_ATT_DOT_KOSAC": KOSACDataset,
    "KOSAC_LSTM_ATT_DOT_ML": KOSACDataset,

    "LSTM_KNU": KNUDataset,
    "LSTM_ATT_KNU": KNUDataset,
    "LSTM_ATT_v2_KNU": KNUDataset,
    "LSTM_ATT_DOT_KNU": KNUDataset,
    "KOSAC_LSTM_ATT_DOT_ML": KNUDataset,

    "CHAR_KOELECTRA": CharBaseDataset,

    "EMB2_LSTM": BaseDataset,
    "EMB1_LSTM2" : BaseDataset
}
