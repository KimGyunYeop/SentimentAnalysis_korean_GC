import pandas as pd

base_f = open('../data/nsmc/ratings_train.txt', encoding='utf-8-sig')
f4_f = open('../crawling/4flix_Reviews.txt', encoding='utf-8-sig')
kino_f = open('../crawling/kinolights_Reviews.txt', encoding='utf-8-sig')
ip_f = open('../crawling/sports_interpark_Reviews.txt', encoding='utf-8-sig')
watcha_f = open('../crawling/watcha_Reviews.txt', encoding='utf-8-sig')

list_all = base_f.readlines()
print('len all', len(list_all))

start = 20000000
idx_idx = 0
# remove duplications from '4flix_Reviews.txt' and convert to string list
f4_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = f4_f.readlines()
for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    a = {"review_id" : start + idx_idx, "review" : sp[1], "rating" : int(sp[2])}
    idx_idx+=1
    f4_df = f4_df.append(a, ignore_index=True)
print(len(f4_df))
f4_df = f4_df.drop_duplicates(subset = ['review'])
print(len(f4_df))
# list to array
list_f4 = f4_df.values.tolist()
arr_f4 = []
for i in range(len(list_f4)):
    arr_f4.append(str(list_f4[i][0])+'\t'+str(list_f4[i][1])+'\t'+str(list_f4[i][2])+'\n')
print('f4  length', len(arr_f4))

# remove duplications from 'kinolights_Reviews.txt' and convert to string list
kino_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = kino_f.readlines()
for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    a = {"review_id" : start + idx_idx, "review" : sp[2], "rating" : int(sp[3])}
    idx_idx += 1
    kino_df = kino_df.append(a, ignore_index=True)
print(len(kino_df))
kino_df = kino_df.drop_duplicates(subset = ['review'])
print(len(kino_df))
# list to array
list_kino = kino_df.values.tolist()
arr_kino = []
for i in range(len(list_kino)):
    arr_kino.append(str(list_kino[i][0])+'\t'+str(list_kino[i][1])+'\t'+str(list_kino[i][2])+'\n')
print('kino length', len(arr_kino))

# remove duplications from 'watcha_Reviews.txt' and convert to string list
watcha_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = watcha_f.readlines()
for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    a = {"review_id" : start + idx_idx, "review" : sp[2], "rating" : int(sp[3])}
    idx_idx += 1
    watcha_df = watcha_df.append(a, ignore_index=True)
print(len(watcha_df))
watcha_df = watcha_df.drop_duplicates(subset = ['review'])
print(len(watcha_df))
# list to array
list_wc = watcha_df.values.tolist()
arr_wc = []
for i in range(len(list_wc)):
    arr_wc.append(str(list_wc[i][0])+'\t'+str(list_wc[i][1])+'\t'+str(list_wc[i][2])+'\n')
print('watcha length', len(arr_wc))

# convert format of sportsReview
ip_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = ip_f.readlines()
#new_l = ['\t' if lines=='\n' else line for line in lines]

# concat sentence of same review
new_lines = []
print(len(lines))
check_loc = 0 # 1: line done, 0: line continue
s =''

for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    if sp[-1][0] == '0' or sp[-1][0] == '1' or sp[-1][0] == '2' or sp[-1][0] == '3' or sp[-1][0] == '4' or sp[-1][0] == '5' or sp[-1][0] == '6' or sp[-1][0] == '7' or sp[-1][0] == '8' or sp[-1][0] == '9' or sp[-1][0] == '10' :
        if len(sp[-1]) == 3 or len(sp[-1]) == 4:
            new_lines.append(s + lines[i])
            s = ''
        else:
            lines[i] = lines[i].replace('\n', ' ')
            s+=lines[i]
    else:
        lines[i] = lines[i].replace('\n', ' ')
        s += lines[i]
print('new', len(new_lines))

for i in range(1, len(new_lines)):
    sp = new_lines[i].split('\t')
    a = {"review_id": start + idx_idx, "review": sp[2], "rating": int(sp[3])}
    idx_idx += 1
    ip_df = ip_df.append(a, ignore_index=True)
print(len(ip_df))
ip_df = ip_df.drop_duplicates(subset = ['review'])
print(len(ip_df))
# list to array
list_ip = ip_df.values.tolist()
arr_ip = []
for i in range(len(list_ip)):
    arr_ip.append(str(list_ip[i][0])+'\t'+str(list_ip[i][1])+'\t'+str(list_ip[i][2])+'\n')
print('interpark length', len(arr_ip))

# append nsmc, 4flix,kino, watcha, interpark list
list_all = list_all + arr_f4 + arr_kino + arr_wc + arr_ip
print(len(list_all))
f = open("all_Reviews.txt", 'w', encoding='utf-8-sig')
for i in range(len(list_all)):
    f.write(str(list_all[i]))

# file close
base_f.close()
f4_f.close()
kino_f.close()
ip_f.close()
watcha_f.close()
