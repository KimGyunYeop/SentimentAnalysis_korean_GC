import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pickle
import os


all_urls = []

MAX_LEVEL = 3

def search_node(url, sentiment, level):
    #get now word's dict homepage
    global driver

    try:
        driver.get(url)
        WebDriverWait(driver, 2).until(lambda x: url.split("/")[-1] in x.page_source)
    except:
        driver.close()
        driver = webdriver.Chrome(chromeDriver)
        search_node(url, sentiment, level)
        return 0

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

base_lex_list = list()

origin_lex = pd.read_csv("KNU_lexicon.tsv",sep="\t")
origin_lex = origin_lex[~origin_lex['ngram'].str.contains(' ')]

if os.path.isfile("base_lex_list.list"):
    with open("base_lex_list.list", 'rb') as f:
        base_lex_list = list(pickle.load(f))
    for data in base_lex_list:
        all_urls.append(data["url"])
else:
    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    for idx in origin_lex.index:
        ori_word = origin_lex.at[idx,"ngram"]
        sent = origin_lex.at[idx,"sentiment"]
        driver.get("https://ko.dict.naver.com/#/search?query="+ori_word)
        try:
            WebDriverWait(driver, 1).until(lambda x: ori_word in x.page_source)
        except:
            continue

        bs = BeautifulSoup(driver.page_source, "html.parser")
        content = bs.find("div",{"id":"searchPage_entry"})
        if content is None:
            continue
        words = content.findAll("div",{"class":"row"})
        for word in words:
            if similar(word.find("button",{"class":"unit_add_wordbook _btn_add_wordbook"}).get("entryname"),ori_word) > 0.7:
                url = word.find("div",{"class":"origin"}).find("a").get("href")
                if url in all_urls:
                    continue

                data = dict()
                data["word"] = word.find("button",{"class":"unit_add_wordbook _btn_add_wordbook"}).get("entryname")
                data["sentiment"] = sent
                data["url"] = url
                base_lex_list.append(data)
                all_urls.append(data["url"])
                #if word.find("dl",{"class":"synonym_info"}) is not None:


    with open("base_lex_list.list", 'wb') as f:
        pickle.dump(base_lex_list, f)

print(base_lex_list)

sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])
for data in base_lex_list:
    if data["sentiment"] > 0 :
        data["sentiment"] = 1
    else:
        data["sentiment"] = -1
    search_node(baselink + data["url"], data["sentiment"], 0)

driver.close()
sent_dc.to_csv("result_add_lex.csv", encoding='utf-8-sig')
