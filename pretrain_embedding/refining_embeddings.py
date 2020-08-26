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
    def __init__(self, dic, w2v):
        super(REFINEEMB, self).__init__()
        self.w2v = w2v
        self.dic = dic
        vectors = []
        for word in dic.keys():
            try:
                vectors.append(w2v.wv.word_vec(word))
            except:
                continue
        self.vector_parameters = nn.Parameter(torch.tensor(vectors, requires_grad = True,dtype=torch.float))
        self.softmax = nn.Softmax()
    def distance(self, x, y):
        return torch.sum((x-y)*(x-y),dim=-1)
    def loss(self,parameters,neighbors):
        weight = torch.FloatTensor([1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10]).repeat(len(parameters),1)
        return torch.sum(weight * self.softmax(self.distance(parameters, neighbors)),dim=-1)

    def forward(self, neighbors):
        batch_parameters = self.vector_parameters.unsqueeze(1).repeat(1,10,1).view(-1,10,200)
        total_loss = torch.sum(self.loss(batch_parameters,neighbors)).tolist()

        return torch.tensor([total_loss/1000], requires_grad = True,dtype=torch.float)

tkn2pol = pickle.load(open(os.path.join('../lexicon','kosac_polarity.pkl'), 'rb'))
tkn2int = pickle.load(open(os.path.join('../lexicon','kosac_intensity.pkl'), 'rb'))
print(tkn2pol)
print(tkn2int)
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
model = REFINEEMB(dic_sentiment2score, word2vec)
model.cuda()
model.to(device)

#optimizer = AdamW(model.parameters(), lr=5e-5)
params_to_update = []
for name, param in model.named_parameters():
    param.requires_grad = True
    params_to_update.append(param)

optimizer = Adam(model.parameters(), lr=5e-5)
model.train()
loss_fn = nn.MSELoss()
ground_truth = torch.FloatTensor([0])
for epoch in range(10):
    optimizer.zero_grad()
    neighbors = torch.FloatTensor(neighbors).to(device)
    loss = model(neighbors)
    loss.backward()
    optimizer.step()
    print("loss : ",loss)
    #print("vectors : ",model.vector_parameters[0])
    print(model.vector_parameters.grad)


