import multiprocessing as mp
from gensim.models import KeyedVectors
from contextlib import closing
import re
import os


reddit_location = r'D:\Datasets\IsItCorrect\french_corpus\parsed_text.txt'
with closing(open(reddit_location, 'r', encoding='utf8')) as file:
    list_lines = file.readlines()

total = 12
list_lines = " ".join(list_lines)
list_lines = pattern.findall(list_lines)

def divide_list(i):
    global list_lines
    global total
    return list_lines[int(i*len(list_lines)/total): int((i+1)*len(list_lines)/total)]

def getdata(list_lines, num):
    clean_reddit_location = r'D:\Datasets\IsItCorrect\french_corpus\reddit_clean{}.txt'.format(num)
    punct_list = ['.', ',', '!', '?', '"']
    unknown_token = '<UNK>'
    with closing(open(clean_reddit_location, 'w')) as file:
        for line in list_lines:
            if len(line) == 1:
                continue
            line = line[7:-5]
            line = line.lower()
            line = line.split(" ")
            for i in range(len(line)):
                if line[i] in punct_list:
                    line[i] = line[i][-1]
            break_int = 0
            for i in range(len(line)):
                if line[i] not in [start_token, end_token]:
                    if '-' in line[i]:
                        temp = line[i].split("-")
                        del line[i]
                        for j in range(len(temp)):
                            line.insert(i+j, temp[j])
                    if '’' in line[i]:
                        temp = line[i].split("’")
                        del line[i]
                        for j in range(len(temp)):
                            line.insert(i+j, temp[j])
                    if line[i] not in vocab:
                        line[i] = unknown_token
                        break_int += 1
            if break_int > 5:
                continue
            else:
                line.insert(0, start_token)
                line.append(end_token)
                line = (" ").join(line)
                try:
                    file.write(line)
                except UnicodeEncodeError:
                    continue

if __name=='__main__':
    global list_lines
    start_token = '<START>'
    end_token = '<END>'
    pattern = re.compile('<START>.+?<END>')
    model_location = r'D:\Datasets\IsItCorrect\fra_word_embedding_100\model.bin'
    model = KeyedVectors.load_word2vec_format(model_location,
                                              binary=True)
    word_vectors = model.wv
    vocab = list(word_vectors.vocab.keys())

    divided = [divide_list(i) for i in range(total)]
    list_lines.clear()

    p0 = mp.Process(target=getdata, args = (divided[0], 0, ))
    p1 = mp.Process(target=getdata, args=(divided[1], 1, ))
    p2 = mp.Process(target=getdata, args=(divided[2], 2, ))
    p3 = mp.Process(target=getdata, args=(divided[3], 3, ))
    p4 = mp.Process(target=getdata, args=(divided[4], 4, ))
    p5 = mp.Process(target=getdata, args=(divided[5], 5, ))
    p6 = mp.Process(target=getdata, args=(divided[6], 6, ))
    p7 = mp.Process(target=getdata, args=(divided[7], 7, ))
    p8 = mp.Process(target=getdata, args=(divided[8], 8, ))
    p9 = mp.Process(target=getdata, args=(divided[9], 9, ))
    p10 = mp.Process(target=getdata, args=(divided[10], 10, ))
    p11 = mp.Process(target=getdata, args=(divided[11], 11, ))

    process_list = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

