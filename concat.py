import pandas as pd

base_f = open('../data/nsmc/ratings_train.txt', encoding='utf-8-sig')
f4_f = open('../crawling/4flix_Reviews.txt', encoding='utf-8-sig')
kino_f = open('../crawling/kinolights_Reviews.txt', encoding='utf-8-sig')
ip_f = open('../crawling/sports_interpark_Reviews.txt', encoding='utf-8-sig')
watcha_f = open('../crawling/watcha_Reviews.txt', encoding='utf-8-sig')
ani_f =  open('../crawling/RA/aniplus_Reviews.txt', encoding='utf-8-sig')
yn_f = open('../crawling/RA/youtube_comments_negative.txt', encoding='utf-8-sig')
yp_f = open('../crawling/RA/youtube_comments_positive.txt', encoding='utf-8-sig')

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

# remove duplications from 'RA/aniplus_Reviews.txt' and convert to string list
ani_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = ani_f.readlines()
for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    a = {"review_id" : start + idx_idx, "review" : sp[1], "rating" : int(sp[2])}
    idx_idx += 1
    ani_df = ani_df.append(a, ignore_index=True)
print(len(ani_df))
ani_df = ani_df.drop_duplicates(subset = ['review'])
print(len(ani_df))
# list to array
list_ani = ani_df.values.tolist()
arr_ani = []
for i in range(len(list_ani)):
    arr_ani.append(str(list_ani[i][0])+'\t'+str(list_ani[i][1])+'\t'+str(list_ani[i][2])+'\n')
print('aniplus length', len(arr_ani))

# remove duplications from 'RA/youtube_comments_negative.txt' and convert to string list
yn_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = yn_f.readlines()
for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    a = {"review_id" : start + idx_idx, "review" : sp[1], "rating" : 0}
    idx_idx += 1
    yn_df = yn_df.append(a, ignore_index=True)
print(len(yn_df))
yn_df = yn_df.drop_duplicates(subset = ['review'])
print(len(yn_df))
# list to array
list_yn = yn_df.values.tolist()
arr_yn = []
for i in range(len(list_yn)):
    arr_yn.append(str(list_yn[i][0])+'\t'+str(list_yn[i][1])+'\t'+str(list_yn[i][2])+'\n')
print('youtube negative length', len(list_yn))

# remove duplications from 'RA/youtube_comments_positive.txt' and convert to string list
yp_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
lines = yp_f.readlines()
for i in range(1, len(lines)):
    sp = lines[i].split('\t')
    a = {"review_id" : start + idx_idx, "review" : sp[1], "rating" : 1}
    idx_idx += 1
    yp_df = yp_df.append(a, ignore_index=True)
print(len(yp_df))
yp_df = yp_df.drop_duplicates(subset = ['review'])
print(len(yp_df))
# list to array
list_yp = yp_df.values.tolist()
arr_yp = []
for i in range(len(list_yp)):
    arr_yp.append(str(list_yp[i][0])+'\t'+str(list_yp[i][1])+'\t'+str(list_yp[i][2])+'\n')
print('youtube positive length', len(list_yp))

# append nsmc, 4flix,kino, watcha, interpark, aniplus list
list_all = list_all + arr_f4 + arr_kino + arr_wc + arr_ip + arr_ani + arr_yn +  arr_yp
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
ani_f.close()
yn_f.close()
yp_f.close()
