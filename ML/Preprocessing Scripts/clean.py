from gensim.models import KeyedVectors
from contextlib import closing
import re

start_token = '<START>'
end_token = '<END>'
unknown_token = '<UNK>'

pattern = re.compile('<START>.+?<END>')
punct_list = ['.', ',', '!', '?', '"']

model_location = r'D:\Datasets\IsItCorrect\fra_word_embedding_100\model.bin'
model = KeyedVectors.load_word2vec_format(model_location,
                                          binary=True)

word_vectors = model.wv
vocab = list(word_vectors.vocab.keys())


def do_the_cleaning(file_location, clean_file_location):
    if 'stories' in file_location:
        with closing(open(file_location, 'r', encoding='8')) as file:
            lines = file.readlines()
            list_lines = pattern.findall(lines[0])
    else:
        with closing(open(file_location, 'r', encoding='utf16')) as file:
            list_lines = file.readlines()
            for i in range(len(list_lines)):
                list_lines[i] = list_lines[i][:-1]
            list_lines = " ".join(list_lines)
            list_lines = list_lines.split(".")
            for i in range(len(list_lines)):
                list_lines[i] = start_token + " " + list_lines[i] + " " + end_token

    with closing(open(clean_file_location, 'w')) as file:
        for line in list_lines:
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
                            line.insert(i + j, temp[j])
                    if '’' in line[i]:
                        temp = line[i].split("’")
                        del line[i]
                        for j in range(len(temp)):
                            line.insert(i + j, temp[j])
                    if line[i] not in vocab:
                        line[i] = unknown_token
                        break_int += 1
            if break_int > 7:
                continue
            else:
                line.insert(0, start_token)
                line.append(end_token)
                line = " ".join(line)
                try:
                    file.write(line)
                except UnicodeEncodeError:
                    continue


clean_stories_location = r'D:\Datasets\IsItCorrect\french_corpus\stories_clean.txt'
stories_location = r'D:\Datasets\IsItCorrect\french_corpus\stories.txt'

clean_novel_location = r'D:\Datasets\IsItCorrect\french_corpus\novel_clean.txt'
novel_location = r'D:\Datasets\IsItCorrect\french_corpus\novel.txt'

do_the_cleaning(stories_location, clean_stories_location)
do_the_cleaning(novel_location, clean_novel_location)