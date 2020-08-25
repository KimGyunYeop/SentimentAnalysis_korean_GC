from konlpy.tag import Okt
import glob
import re
file_list = glob.glob("../corpus/*.txt")
print(file_list)
okt = Okt()
for file_name in file_list[0:1]:
    print("file name : ",file_name)
    my_file = open(file_name, "r",encoding="utf16")
    sentences = my_file.readlines()
    for sentence in sentences:
        print(sentence)
        m = re.sub(r'<[a-zA-Z0-9"= ]*>', ' ', sentence)
        m = re.sub(r'</[a-zA-Z0-9"= ]*>', ' ', m)
        m = re.sub(r'<[a-zA-Z0-9"= ]*/>', ' ', m)
        print(m)

        found_sentence = ""
        print(found_sentence)