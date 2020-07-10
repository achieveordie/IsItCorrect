import multiprocessing as mp
from gensim.models import KeyedVectors
from contextlib import closing
import re
import time

model_location = r'D:\Datasets\IsItCorrect\fra_word_embedding_100\model.bin'
model = KeyedVectors.load_word2vec_format(model_location,
                                              binary=True)
word_vectors = model.wv
vocab = list(word_vectors.vocab.keys())

reddit_location = r'D:\Datasets\IsItCorrect\testing.txt'
with closing(open(reddit_location, 'r', encoding='utf8')) as file:
    list_lines = file.readlines()

total = 6
bunchsize = 10000
start_token = '<START>'
end_token = '<END>'
unknown_token = '<UNK>'

list_lines = " ".join(list_lines)
pattern = re.compile('<START>.+?<END>')
list_lines = pattern.findall(list_lines)


def divide_list(i):

    return list_lines[int(i * len(list_lines) / total): int((i + 1) * len(list_lines)-1 / total)]


def getdata(list_lines, num):
    a = time.time()
    print("Process {} started".format(num))
    clean_reddit_location = r'D:\Datasets\IsItCorrect\french_corpus\reddit_clean{}.txt'.format(num)
    punct_list = ['.', ',', '!', '?', '"']
    unknown_token = '<UNK>'
    with closing(open(clean_reddit_location, 'w')) as file:
        bunch = []
        for line in list_lines:
            if len(line) == 1:
                continue
            line = line[7:-5].lower().split(" ")
            for i in range(len(line)):
                if line[i] in punct_list:
                    line[i] = line[i][:-1]
            break_int = 0
            for i in range(len(line)):
                if line[i] not in [start_token, end_token]:
                    if '-' in line[i]:
                        temp = line[i].split("-")
                        del line[i]
                        for j in range(len(temp)):
                            line.insert(i + j, temp[j])
                    if "'" in line[i]:
                        temp = line[i].split("'")
                        del line[i]
                        for j in range(len(temp)):
                            line.insert(i + j, temp[j])
                    if line[i] not in vocab:
                        line[i] = unknown_token
                        break_int += 1
            if break_int > 4:
                continue
            else:
                line.insert(0, start_token)
                line.append(end_token)
                line = " ".join(line)
                bunch.append(line)
                if len(bunch) == bunchsize:
                    try:
                        file.write(" ".join(bunch))
                        bunch = []
                    except UnicodeEncodeError:
                        bunch = []
                        continue
    b = time.time()
    print("Done for Process number {}, with time {}".format(num, b-a))


if __name__ == '__main__':

    divided = [divide_list(i) for i in range(total)]
    list_lines.clear()
    model = None
    word_vectors = None

    p0 = mp.Process(target=getdata, args=(divided[0], 0,))
    p1 = mp.Process(target=getdata, args=(divided[1], 1,))
    p2 = mp.Process(target=getdata, args=(divided[2], 2,))
    p3 = mp.Process(target=getdata, args=(divided[3], 3,))
    p4 = mp.Process(target=getdata, args=(divided[4], 4,))
    p5 = mp.Process(target=getdata, args=(divided[5], 5,))

    process_list = [p0, p1, p2, p3, p4, p5]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
