import pandas as pd
import numpy as np
from konlpy.tag import Kkma, Komoran, Mecab, Hannanum, Twitter, Okt

SNU_df = pd.read_csv("polarity.csv", encoding="utf8")
print(SNU_df)
'''
KNU_df = pd.DataFrame(columns=["ngram","sentiment"])
with open("SentiWord_Dict.txt","r",encoding="utf8") as fp:
    count = 0
    for i in fp.readlines():
        data = i.split()
        if len(data)>2:
            KNU_df.loc[count,"ngram"] = " ".join(data[:-1])
        else:
            KNU_df.loc[count, "ngram"] = data[0]
        KNU_df.loc[count,"sentiment"] = data[-1]
        count+=1

print(KNU_df)
KNU_df.to_csv("KNU_lexicon.tsv",encoding="utf8",sep="\t", index=False)

'''
KNU_df = pd.read_csv("KNU_lexicon.tsv",encoding="utf8",sep="\t")

#pos_tagger = Kkma()
#print(pos_tagger.pos(u"함찬"))
#KNU_df["ngram"].apply(pos_tagger.pos)
print(KNU_df)

data = ";".join(SNU_df["ngram"]).split(";")
data = list(map((lambda x : list(x.split("/"))), data))

pos = sorted(set(np.array(data)[:,-1]))
kkma = sorted(set(Kkma().tagset.keys()))
komoran = sorted(set(Komoran().tagset.keys()))
hannanum = sorted(set(Hannanum().tagset.keys()))

print(len(pos),"+",len(kkma),"=",len(pos+kkma)-len(set(pos+kkma)))
print(len(pos),"+",len(komoran),"=",len(pos+komoran)-len(set(pos+komoran)))
print(len(pos),"+",len(hannanum),"=",len(pos+hannanum)-len(set(pos+hannanum)))