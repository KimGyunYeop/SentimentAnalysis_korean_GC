import pickle
import os
import pandas as pd
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from transformers import AdamW
from torch.optim import Adam
import numpy as np

class REFINEEMB(nn.Module):
    def __init__(self, dic, w2v,device):
        super(REFINEEMB, self).__init__()
        self.w2v = w2v
        self.device = device
        self.dic = dic
        vectors = []
        for word in dic.keys():
            try:
                vectors.append(w2v.wv.word_vec(word))
            except:
                continue
        self.vector_parameter = nn.Parameter(torch.tensor(vectors).t(),requires_grad = True)
        self.linear = nn.Linear(len(vectors), 200, bias=False)
        self.linear.weight = self.vector_parameter
        self.softmax = nn.Softmax(dim=-1)
        self.weight = torch.FloatTensor([1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10]).repeat(len(vectors),1)

    def distance(self, x, y):
        return torch.sum(torch.sub(x,y).mul(2),dim=-1)
    def loss(self,result,neighbors):
        result = torch.sum(self.weight.mul(self.softmax(self.distance(result, neighbors))),dim=-1)
        return result

    def forward(self, ones_data,neighbors):
        result = self.linear(ones_data).unsqueeze(1).repeat(1,10,1)
        total_loss = self.loss(result,neighbors).sum()
        return total_loss

tkn2pol = pickle.load(open(os.path.join('../lexicon','kosac_polarity.pkl'), 'rb'))
tkn2int = pickle.load(open(os.path.join('../lexicon','kosac_intensity.pkl'), 'rb'))
pol2idx = ['None','NEG', 'COMP', 'NEUT', 'POS']
int2idx = ['None','Low', 'Medium', 'High']
dict_pol2idx = {y:x for x,y in enumerate(pol2idx)}
dict_int2idx = {y:x for x,y in enumerate(int2idx)}
dic_sentiment2score = {tokens:(dict_pol2idx[pol] * dict_int2idx[int]) for (tokens,pol),int in zip(tkn2pol.items(),tkn2int.values())}

#word2vec
word2vec = Word2Vec.load('word2vec.model')
error_count = 0
neighbors =  []
for word, score in dic_sentiment2score.items():
    try:
        neighbor = word2vec.wv.similar_by_word(word, topn=10)
        neighbor_score = {}
        for word,_ in neighbor:
            try:
                neighbor_score[word] = dic_sentiment2score[word]
            except:
                neighbor_score[word] = 0
                continue
        neighbor = [word2vec.wv.word_vec(k) for k, _ in sorted(neighbor_score.items(), key=lambda item: item[1])]
        neighbors.append(neighbor)
    except:
        error_count+=1
        continue
print(error_count)
device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
model = REFINEEMB(dic_sentiment2score, word2vec,device)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1)
for param in model.parameters():
    param.requires_grad = True
#optimizer = Adam(model.parameters(),lr=10)
model.train()

print(model)
for epoch in range(10):
    optimizer.zero_grad()
    neighbors = torch.tensor(neighbors, dtype=torch.float).to(device)
    tmp_data = torch.ones(len(neighbors), len(neighbors), requires_grad=True).to(device)
    loss = model(tmp_data,neighbors)
    print("loss : ", loss)
    loss.backward(create_graph=True)
    optimizer.step()


