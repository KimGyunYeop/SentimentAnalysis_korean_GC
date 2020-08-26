import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time

all_urls = []

MAX_LEVEL = 5

def search_node(url, sentiment, level):
    #get now word's dict homepage
    driver.get(url)
    time.sleep(1)
    bs = BeautifulSoup(driver.page_source, "html.parser")

    #get word in metadata
    word_meta = bs.find("meta",{"property":"og:title"})
    cur_word = word_meta.get("content").split("\'")[1]
    print(cur_word, level)
    global sent_dc, all_urls
    sent_dc = sent_dc.append(pd.Series([cur_word, sentiment, level], index=sent_dc.columns), ignore_index=True)

    # 활용어
    dl = bs.find_all("dl")
    ct = 0
    for d in dl:
        if d.get('class')[0] == 'entry_conjugation':
            ct = 1
    if ct == 1:
        aps = bs.find("dl", {"class": "entry_conjugation"}).find("dd", {"class": "cont"}).find("ul", {
            "class": "tray"}).find_all("li", {"class": "item"})
        for ap in aps:
            word = ap.find("span", {"class" : "word"}).text.strip()
            # two use cases
            if '(' in word:
                sent_dc = sent_dc.append(pd.Series([word[:word.find('(')], sentiment,level], index=sent_dc.columns), ignore_index=True)
                sent_dc = sent_dc.append(pd.Series([word[word.find('(')+1:word.find(')')], sentiment, level], index=sent_dc.columns), ignore_index=True)
            else:
                sent_dc = sent_dc.append(pd.Series([word, sentiment,level], index=sent_dc.columns), ignore_index=True)

    if level >= MAX_LEVEL:
        return 0

    antonym_list = []
    synonym_list = []

    #find next word's url and store in antonym_list, synonym_list
    word_box = bs.find("div",{"class":"slides_content _slides_content _visible"})
    if word_box is None:
        return 0
    box_list = word_box.findAll("div")
    for tmp_box in box_list:
        if "synonym" in tmp_box.get("class"):
            synonym_list_box = tmp_box.findAll("em")
            for i in synonym_list_box:
                url = i.find("a",{"class":"blank"}).get("href")
                if url not in all_urls:
                    synonym_list.append(url)
                    all_urls.append(url)

        if "antonym" in tmp_box.get("class"):
            antonym_list_box = tmp_box.findAll("em")
            for i in antonym_list_box:
                url = i.find("a",{"class":"blank"}).get("href")
                if url not in all_urls:
                    antonym_list.append(url)
                    all_urls.append(url)

    #print(antonym_list,synonym_list)

    #find next nodes word
    for i in antonym_list:
        search_node(baselink+i, -(sentiment), level+1)
    for i in synonym_list:
        search_node(baselink+i, sentiment, level+1)


baselink = "https://ko.dict.naver.com/"
chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)

'''sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/61a8e24d442d4b47a7a241797f968195"
all_urls = [first_word_url]
try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_예쁘다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 흥미롭다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/b9fbbe6ab1734c93ba2213b4ac22fbba"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_흥미롭다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 행복하다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/b5ae4f305c8942fc9b20537f1bcfa535"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_행복하다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 유쾌하다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/f5d7898240de45c5b02ab1a0b0f62998"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_유쾌하다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 성공하다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/c4919042e58849cb905680e02d9c7274"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_성공하다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 잘하다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/2ab293623c9d443ab8d7a5e38d2df41d"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_잘하다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 훌륭하다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/57b2a8ad9671484c835ade92aa79e2f8"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_훌륭하다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 좋다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/6cbf564655854b4d8225477320c63e7a"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_좋다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 재미없다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/c0f56a6360ef492092dc16b9bb6937e5"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_재미없다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]'''
# 화나다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/4a0a2f1d4e514fe98051ec9a275738d7"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

all_urls = [first_word_url]
print(sent_dc)
sent_dc.to_csv("result_화나다.csv", encoding='utf-8-sig')

# 언짢다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/68e26f5324214fb6a829616d70f1a261"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_언짢다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 형편없다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/806b30d3bb0442a091f51de3607c7686"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_형편없다.csv", encoding='utf-8-sig')
all_urls = []
# 괴롭다
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/37cd5ce39cbd4f97a07ed6d6e4000ebf"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_괴롭다.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 여기서부턴 단어
# 사랑
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/102e6fab94954bb1a61c70b70dc4d781"

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_사랑.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 최악
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/ec82b9ddab10453cb74a7802b245bbb5"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_최악.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 혐오
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/e51820bdd27a4733ab00a0a221f5b44a"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_혐오.csv", encoding='utf-8-sig')
all_urls = [first_word_url]
# 구닥다리
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
first_word_url = "#/entry/koko/f1da4cfceed64a64a1ecd24034f88401"

try:
    search_node(baselink+first_word_url, -1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_구닥다리.csv", encoding='utf-8-sig')
driver.close()