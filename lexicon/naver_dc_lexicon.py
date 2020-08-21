from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time

import random

urls = []
# 현재 단어와 유의어인지 반의어인지
rev = []

results = pd.DataFrame()
words = []
sentiments = []
adj_sents = []
tests = []

test = 0

searched = []

chromeDriver = "C:\\Users\\parksoyoung\\Downloads\\chromedriver_win32\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
driver.get('https://ko.dict.naver.com/#/entry/koko/1b3ddffea9e44296a09842b0776d6997')
time.sleep(3)

# 시작 단어가 꽃답다(긍정)이기 때문에
senti = 1

# 검색한 단어
searched.append("꽃답다")

# 초기 단어
words.append("꽃답다")
sentiments.append(senti)
tests.append(test)

li = ['', '']

source = driver.page_source
soup = BeautifulSoup(source, "html.parser")

# 한 단어 크롤링
def oneWord(sent):
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")
    # 활용 어간
    dl = soup.find_all("dl")
    ct = 0
    for d in dl:
        if d.get('class')[0] == 'entry_conjugation':
            ct = 1
    aps = []
    if ct == 1:
        aps = soup.find("dl", {"class": "entry_conjugation"}).find("dd", {"class": "cont"}).find("ul", {
            "class": "tray"}).find_all("li", {"class": "item"})
    for ap in aps:
        word = ap.find("span", {"class": "u_word_dic"}).text.strip()
        words.append(word)
        sentiments.append(sent)
        tests.append(test)

    s = soup.find("input", {"id":"hiddenQuery"})
    searched.append(s.get("value"))

    # 유의어, 반의어
    dc = soup.find("div", {"class": "slides_content _slides_content _visible"})
    div = dc.find_all("div")
    r = random.randint(1, 2)
    # 초기값: 유의어
    for d in div:
        if d.get("class")[0] == 'synonym':
            li = d.get("class")
    for d in div:
        # 유의어, 반의어 중에 랜덤으로 누르기
        if r == 1:
            if d.get("class")[0] == 'synonym':
                li = d.get("class")
        elif r == 2:
            if d.get("class")[0] == 'antonym':
                li = d.get("class")

        if d.get("class")[0] != 'key_word':
            em = d.find_all("em")
            if d.get("class")[0] == 'antonym':
                # 반대말 단어들
                sent *= -1
            ym = d.get("class")
            i=0
            for e in em:
                word = e.find("a").text.strip()
                # 숫자 붙으면 지워주기
                if word[-1] == '1' or word[-1] == '2' or word[-1] == '3' or word[-1] == '4' or word[-1] == '5':
                    word = word[0:-1]
                words.append(word)
                sentiments.append(sent)
                tests.append(test)
                # 유의어/반의어의 활용 어간
                concat_ym = ym[0] + ' ' + ym[1]
                i+=1
                driver.find_element_by_xpath(
                    '//*[@id="container"]/div[@id="content"]/div[@class="section section_thesaurus _section_thesaurus"]'
                    '/div[@class="component_thesaurus"]/div[@class="thesaurus_inner"]/div[@class="map"]'
                    '/div[@class="slides"]//div[@class="slides_container _slides_container"]'
                    '/div[@class="slides_content _slides_content _visible"]/div[@class="' + concat_ym +
                    '"]/em[@class="num' + str(i) + '"]'
                    '/a[2]').click()
                # 가장 최근 연 탭으로
                driver.switch_to.window(driver.window_handles[-1])
                # url 저장
                #print(driver.current_url)
                urls.append(driver.current_url)
                if ym[0] == 'synonym':
                    adj_sents.append(sent)
                else:
                    adj_sents.append(sent*-1)
                time.sleep(3)
                source = driver.page_source
                soup = BeautifulSoup(source, "html.parser")
                # 활용 어간
                dl = soup.find_all("dl")
                ct = 0
                for d in dl:
                    if d.get('class')[0] == 'entry_conjugation':
                        ct = 1
                aps = []
                if ct == 1:
                    aps = soup.find("dl", {"class": "entry_conjugation"}).find("dd", {"class": "cont"}).find("ul", {
                        "class": "tray"}).find_all("li", {"class": "item"})
                for ap in aps:
                    word = ap.find("span", {"class": "u_word_dic"}).text.strip()
                    words.append(word)
                    sentiments.append(sent)
                    tests.append(test)

                # 현재 탭 닫고 맨 처음 탭으로 변경 (0번 탭)
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
    return li

time.sleep(3)

def start(link, senti):
    driver.get(link)
    time.sleep(3)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")
    oneWord(senti)

# 꽃답다, 긍정
#start('https://ko.dict.naver.com/#/entry/koko/1b3ddffea9e44296a09842b0776d6997', 1)
senti = 1
urls.append('https://ko.dict.naver.com/#/entry/koko/1b3ddffea9e44296a09842b0776d6997')
rev.append(1)
adj_sents.append(1)

while True:
    prev = len(words)
    urls = list(set(urls))
    print(len(urls))
    for i in range(len(urls)):
        results = pd.DataFrame()
        #print(urls[i])
        start(urls[i], adj_sents[i])
        if prev == len(words):
            break
        results['word'] = words
        results['sentiment'] = sentiments
        results['test'] = tests
        print(len(results))
        uni = results.drop_duplicates(subset=['word'])
        print(len(uni))
        results.to_csv('naver_dc_꽃답다.txt', encoding="utf8", sep="\t")
        uni.to_csv('naver_dc_꽃답다_uni.txt', encoding="utf8", sep="\t")
