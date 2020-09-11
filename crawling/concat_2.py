import pandas as pd

all_f = open("../crawling/all_Reviews.txt", encoding='utf-8-sig')
list_all = all_f.readlines()
start = 20000000 + len(list_all)
all_f.close()
all_f = open("../crawling/all_Reviews.txt", 'a', encoding='utf-8-sig')
idx_idx = 0
'''
sl_f = open('../crawling/RA/samsungLions_Reviews.txt', encoding='utf-8-sig')
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
sl_f.close()

dk_f = open('../crawling/sports_auc.txt', encoding='utf-8-sig')
# convert format of sportsReview
dk_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = dk_f.readlines()
#new_l = ['\t' if lines=='\n' else line for line in lines]

# concat sentence of same review
new_lines = []
print(len(lines))
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
    dk_df = dk_df.append(a, ignore_index=True)
print(len(dk_df))
dk_df = dk_df.drop_duplicates(subset = ['review'])
print(len(dk_df))
# list to array
list_ip = dk_df.values.tolist()
arr_ip = []
for i in range(len(list_ip)):
    arr_ip.append(str(list_ip[i][0])+'\t'+str(list_ip[i][1])+'\t'+str(list_ip[i][2])+'\n')
print('interpark length', len(arr_ip))

list_all = list_all + arr_ip
print(len(list_all))
f = open("all_Reviews.txt", 'w', encoding='utf-8-sig')
for i in range(len(list_all)):
    f.write(str(list_all[i]))
dk_f.close()
'''
dk_f = open('../crawling/sports_auc_golf.txt', encoding='utf-8-sig')
# convert format of sportsReview
dk_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = dk_f.readlines()
#new_l = ['\t' if lines=='\n' else line for line in lines]

# concat sentence of same review
new_lines = []
print(len(lines))
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
    dk_df = dk_df.append(a, ignore_index=True)
print(len(dk_df))
dk_df = dk_df.drop_duplicates(subset = ['review'])
print(len(dk_df))
# list to array
list_ip = dk_df.values.tolist()
arr_ip = []
for i in range(len(list_ip)):
    arr_ip.append(str(list_ip[i][0])+'\t'+str(list_ip[i][1])+'\t'+str(list_ip[i][2])+'\n')
print('interpark length', len(arr_ip))

list_all = list_all + arr_ip
print(len(list_all))
f = open("all_Reviews.txt", 'w', encoding='utf-8-sig')
for i in range(len(list_all)):
    f.write(str(list_all[i]))
dk_f.close()

all_f.close()

