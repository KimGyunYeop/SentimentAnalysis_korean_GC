import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time

MAX_LEVEL = 5

def search_node(url, sentiment, level):
    #get now word's dict homepage
    driver.get(url)
    time.sleep(0.3)
    bs = BeautifulSoup(driver.page_source, "html.parser")

    #get word in metadata
    word_meta = bs.find("meta",{"property":"og:title"})
    cur_word = word_meta.get("content").split("\'")[1]
    global sent_dc
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
                synonym_list.append(i.find("a",{"class":"blank"}).get("href"))

        if "antonym" in tmp_box.get("class"):
            antonym_list_box = tmp_box.findAll("em")
            for i in antonym_list_box:
                antonym_list.append(i.find("a",{"class":"blank"}).get("href"))

    #print(antonym_list,synonym_list)

    #find next nodes word
    for i in antonym_list:
        search_node(baselink+i, -(sentiment), level+1)
    for i in synonym_list:
        search_node(baselink+i, sentiment, level+1)


baselink = "https://ko.dict.naver.com/"
first_word_url = "#/entry/koko/61a8e24d442d4b47a7a241797f968195"
chromeDriver = "C:\\Users\\parksoyoung\\Downloads\\chromedriver_win32\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
sent_dc = pd.DataFrame(columns=["word",'sentiment',"level"])

try:
    search_node(baselink+first_word_url, 1, 0)
except:
    pass

print(sent_dc)
sent_dc.to_csv("result_예쁘다.csv", encoding='utf-8-sig')

driver.close()
