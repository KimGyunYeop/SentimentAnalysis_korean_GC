import pickle
import os
import pandas as pd
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from transformers import AdamW
from torch.optim import Adam

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
        self.vector_parameters = nn.Parameter(torch.FloatTensor(vectors))
        self.softmax = nn.Softmax()
    def distance(self, x, y):
        return torch.sum((x-y)*(x-y),dim=-1)
    def loss(self,parameters,neighbors):
        weight = torch.FloatTensor([1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10])
        return torch.sum(weight * self.softmax(self.distance(parameters, neighbors)),dim=-1)
    def forward(self, neighbors):
        total_loss =  0
        for i in range(len(neighbors)):
            batch_parameters = self.vector_parameters[i].repeat(10,1)
            total_loss += self.loss(batch_parameters,neighbors[i]).tolist()
        return torch.FloatTensor([total_loss])

tkn2pol = pickle.load(open(os.path.join('../lexicon','kosac_polarity.pkl'), 'rb'))
tkn2int = pickle.load(open(os.path.join('../lexicon','kosac_intensity.pkl'), 'rb'))
print(tkn2pol)
print(tkn2int)
['None', 'POS', 'NEUT', 'COMP', 'NEG']
pol2idx = ['None','NEG', 'COMP', 'NEUT', 'POS']
int2idx = ['None','Low', 'Medium', 'High']
dict_pol2idx = {y:x for x,y in enumerate(pol2idx)}
dict_int2idx = {y:x for x,y in enumerate(int2idx)}
print(dict_pol2idx)
print(tkn2pol.items())
dic_sentiment2score = {tokens:(dict_pol2idx[pol] * dict_int2idx[int]) for (tokens,pol),int in zip(tkn2pol.items(),tkn2int.values())}
print(len(dic_sentiment2score))

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

model = REFINEEMB(dic_sentiment2score, word2vec)
#optimizer = AdamW(model.parameters(), lr=5e-5)
params_to_update = []
for name, param in model.named_parameters():
    param.requires_grad = True
    params_to_update.append(param)

optimizer = Adam([
    {'params': params_to_update, 'weight_decay': 0.1}
], lr=0.001)
model.train()
for epoch in range(10):
    neighbors = torch.FloatTensor(neighbors)
    loss = model(neighbors)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss : ",loss)


