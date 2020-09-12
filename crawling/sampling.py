import pandas as pd
df = pd.read_csv('daumMovie_Reviews.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
pos = df[df['label']==1]
print('daum pos', len(pos))
neg = df[df['label']==0]
print('daum neg', len(neg))

pos_sample = pos.sample(n=15000)
print(len(pos_sample))
neg_sample = neg.sample(n=10000)
print(len(neg_sample))

pos_sample.to_csv('daum_posSample.txt')
neg_sample.to_csv('daum_negSample.txt')

all_df = pd.read_csv('naverMovie_Reviews_2016.txt', sep='\t')
all_df = all_df.drop_duplicates(subset = ['reviews'])
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))
df = pd.read_csv('naverMovie_Reviews_2017.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
all_df = all_df.append(df)
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))
df = pd.read_csv('naverMovie_Reviews_2018.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
all_df = all_df.append(df)
pos = df[df['label']==1]
print(len(pos))
neg = df[df['label']==0]
print(len(neg))
df = pd.read_csv('naverMovie_Reviews_2019.txt', sep='\t')
df = df.drop_duplicates(subset = ['reviews'])
all_df = all_df.append(df)
pos = all_df[all_df['label']==1]
print(len(pos))
neg = all_df[all_df['label']==0]
print(len(neg))

pos_sample = pos.sample(n=15000)
print(len(pos_sample))
neg_sample = neg.sample(n=10000)
print(len(neg_sample))

pos_sample.to_csv('naver_posSample.txt')
neg_sample.to_csv('naver_negSample.txt')

