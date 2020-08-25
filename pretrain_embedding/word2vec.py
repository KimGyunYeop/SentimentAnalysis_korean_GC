from konlpy.tag import Okt
import glob
import re
import pandas as pd

file_list = glob.glob("../corpus/*.txt")
print(file_list)

okt = Okt()
for file_name in file_list:
    print("file name : ",file_name)
    my_file = open(file_name, "r", encoding="utf16")
    sentences = my_file.readlines()
    for sentence in sentences:
        #print('s', sentence)
        m1 = re.sub(r'<[a-zA-Z.0-9"= ]*>', ' ', sentence)
        m2 = re.sub(r'</[a-zA-Z.0-9"= ]*>', ' ', m1)
        m3 = re.sub(r'<[a-zA-Z.0-9"= ]*/>', ' ', m2)
        if 'vocal desc' in m3:
            m4 = re.sub('<vocal desc=', ' ', m3)
            m4 = re.sub('/>', ' ', m4)
        elif 'event desc' in m3:
            m4 = re.sub('<event desc=', ' ', m3)
            idx = m4.index(">")
            m4 = m4[:idx]
        else:
            m4 = m3

        print(m4)

        found_sentence = ""

        print(found_sentence)
