from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
from requests.packages.urllib3.exceptions import MaxRetryError

urls = []
adj_sents = []
url_df = pd.DataFrame()

results = pd.DataFrame()
words = []
sentiments = []
tests = []

test = 0

searched = []

max_depth = 5

chromeDriver = "C:\\Users\\parksoyoung\\Downloads\\chromedriver_win32\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
time.sleep(3)

source = driver.page_source
soup = BeautifulSoup(source, "html.parser")

# 한 단어 크롤링
def oneWord(sent):
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")

    # 원형 저장
    word = soup.find("meta", {"property": "og:title"})
    cur_word = word.get("content").split("\'")[1]
    words.append(cur_word)
    sentiments.append(sent)
    tests.append(test)
    # 활용어
    dl = soup.find_all("dl")
    # 활용어 유무 확인
    ct = 0
    for d in dl:
        if d.get('class')[0] == 'entry_conjugation':
            ct = 1
    aps = []
    if ct == 1:
        aps = soup.find("dl", {"class": "entry_conjugation"}).find("dd", {"class": "cont"}).find("ul", {
            "class": "tray"}).find_all("li", {"class": "item"})
        for ap in aps:
            #print(ap)
            ap = ap.find("span", {"class": "word"})
            word = ap.find("span", {"class": "u_word_dic"}).text.strip()
            words.append(word)
            sentiments.append(sent)
            tests.append(test)
    # 유의어, 반의어
    # 유의어, 반의어 없는지 확인
    all_div = soup.find_all("div")
    if "slides_content _slides_content _visible" in str(all_div):
        dc = soup.find("div", {"class": "slides_content _slides_content _visible"})
        div = dc.find_all("div")
        for d in div:
            if 'synonym' in d.get("class")[0] or 'antonym' in d.get("class")[0]:
                sa = d.get("class")[0]
                em = d.find_all("em")
                for e in em:
                    urls.append(base_link+ e.find("a", {"class": "blank"}).get("href"))
                    time.sleep(3)
                    if sa == 'synonym':
                        ap_sent = sent
                    else:
                        ap_sent = sent*-1
                    adj_sents.append(ap_sent)
    return 0

time.sleep(3)

def start(link, senti):
    driver.get(link)
    time.sleep(3)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")
    oneWord(senti)

# 흥미롭다
base_link = "https://ko.dict.naver.com/"
senti = 1
urls.append('https://ko.dict.naver.com/#/entry/koko/b9fbbe6ab1734c93ba2213b4ac22fbba')
adj_sents.append(1)

url_df = pd.DataFrame()
url_df['url'] = urls
url_df['adj_sent'] = adj_sents
idx = 0
try:
    while True:
        prev = len(words)
        print(idx)
        for i in range(idx, len(url_df)):
            results = pd.DataFrame()
            start(url_df.iloc[i]['url'], url_df.iloc[i]['adj_sent'])
            idx+=1
            df = pd.DataFrame({'url': urls, 'adj_sent': adj_sents})
            url_df = url_df.append(df)
            if test == max_depth or prev == len(words):
                break
            results['word'] = words
            results['sentiment'] = sentiments
            results['test'] = tests
            print('full', len(results))
            uni = results.drop_duplicates(subset=['word'])
            print('unique', len(uni))
            results.to_csv('naver_dc_흥미롭다.txt', encoding="utf8", sep="\t")
            uni.to_csv('naver_dc_흥미롭다_uni.txt', encoding="utf8", sep="\t")
        test += 1
except ConnectionError as ce:
    if (isinstance(ce.args[0], MaxRetryError)):
        pass
