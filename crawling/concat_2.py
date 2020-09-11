import pandas as pd

all_f = open("../crawling/all_Reviews.txt", encoding='utf-8-sig')
list_all = all_f.readlines()
start = 20000000 + len(list_all)
all_f.close()
all_f = open("../crawling/all_Reviews.txt", 'a', encoding='utf-8-sig')
sl_f = open('../crawling/RA/samsungLions_Reviews.txt', encoding='utf-8-sig')

idx_idx = 0
# save only negative data & remove duplications
sl_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = sl_f.readlines()
for i in range(1, len(lines)):
    lines[i] = lines[i].replace('\n', ' ')
    sp = lines[i].split('\t')
    if sp[1] == '0':
        a = {"review_id" : start + idx_idx, "review" : sp[5], "rating" : 0}
        idx_idx += 1
        sl_df = sl_df.append(a, ignore_index=True)

print(len(sl_df))
print(sl_df)
sl_df = sl_df.drop_duplicates(subset = ['review'])
print(len(sl_df))
# list to array
list_sl = sl_df.values.tolist()
arr_sl = []
for i in range(len(list_sl)):
    arr_sl.append(str(list_sl[i][0])+'\t'+str(list_sl[i][1])+'\t'+str(list_sl[i][2])+'\n')
print('samsung lions length', len(list_sl))

list_all = list_all + arr_sl
print(len(list_all))
f = open("all_Reviews.txt", 'w', encoding='utf-8-sig')
for i in range(len(list_all)):
    f.write(str(list_all[i]))

all_f.close()
sl_f.close()
