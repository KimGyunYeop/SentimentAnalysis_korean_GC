import pandas as pd
df = pd.read_csv('daumMovie_Reviews.txt', sep='\t')
print(len(df))
ip_df = df.drop_duplicates(subset = ['label'])
print(len(df))
#print(df[:5])
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))

pos_sample = pos.sample(n=10000)
print(len(pos_sample))
neg_sample = neg.sample(n=10000)
print(len(neg_sample))

pos_sample.to_csv('daum_posSample.txt')
neg_sample.to_csv('daum_negSample.txt')


df = pd.read_csv('naverMovie_Reviews_2016.txt', sep='\t')
print(len(df))
ip_df = df.drop_duplicates(subset = ['label'])
print(len(df))
#print(df[:5])
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))
df = pd.read_csv('naverMovie_Reviews_2017.txt', sep='\t')
print(len(df))
ip_df = df.drop_duplicates(subset = ['label'])
print(len(df))
pos = pos.append(df[df['label']==1])
print('pos', len(pos))
neg = neg.append(df[df['label']==0])
print('neg', len(neg))
df = pd.read_csv('naverMovie_Reviews_2018.txt', sep='\t')
print(len(df))
ip_df = df.drop_duplicates(subset = ['label'])
print(len(df))
pos = pos.append(df[df['label']==1])
print('pos', len(pos))
neg = neg.append(df[df['label']==0])
print('neg', len(neg))
df = pd.read_csv('naverMovie_Reviews_2019.txt', sep='\t')
print(len(df))
ip_df = df.drop_duplicates(subset = ['label'])
print(len(df))
pos = pos.append(df[df['label']==1])
print('pos', len(pos))
neg = neg.append(df[df['label']==0])
print('neg', len(neg))

pos_sample = pos.sample(n=10000)
print(len(pos_sample))
neg_sample = neg.sample(n=10000)
print(len(neg_sample))

pos_sample.to_csv('naver_posSample.txt')
neg_sample.to_csv('naver_negSample.txt')

