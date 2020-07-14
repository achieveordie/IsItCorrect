
from contextlib import closing
import re
import time
import cProfile


def load_textfile():
    t_loadtxt_1 = time.time()
    reddit_location = r'D:\Datasets\IsItCorrect\french_corpus\parsed_text.txt'
    with closing(open(reddit_location, 'r', encoding='utf8')) as file:
        lines = file.readlines()
    t_loadtxt_2 = time.time()
    print("For load_textfile, time is ", (t_loadtxt_2-t_loadtxt_1))
    return lines


def form_re(lines):
    t_formre_1 = time.time()
    lines = lines[0]
    pattern = re.compile('<START>.*?<END>')
    list_lines = pattern.findall(lines)
    t_fromre_2 = time.time()
    print("For form_re, time is ", (t_fromre_2-t_formre_1))
    return list_lines


def clean_punctuations(list_lines):
    t_cleanpun_1 = time.time()
    start_token = '<START>'
    end_token = '<END>'
    clean_list_lines = []
    punct_list = ['.', ',', '!', '?', '"']

    for line in list_lines:
        if line == start_token+end_token:
            continue
        line = line[7:-5].lower().split(" ")
        for i in range(len(line)):
            try:
                if line[i][-1] in punct_list:
                    line[i] = line[i][:-1]
                if line[i][0] in punct_list:
                    line[i] = line[i][:-1]
            except:
                pass

        clean_list_lines.append(" ".join(line))
    t_cleanpun_2 = time.time()
    print("For clean_punctuations, time is ", (t_cleanpun_2-t_cleanpun_1))
    return clean_list_lines


def clean_additional(list_lines):
    t_cleanadd_1 = time.time()
    clean_list = []
    for line in list_lines:
        line = line.split(" ")
        for i in range(len(line)):
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
        clean_list.append(" ".join(line))
    t_cleanadd_2 = time.time()
    print("For clean_additional, time is ", t_cleanadd_2-t_cleanadd_1)
    return clean_list


def write_into_file(list_lines):
    t_write_1 = time.time()
    for i in range(len(list_lines)):
        list_lines[i] = '<START> '+list_lines[i]+' <END>'
    write_location = r'D:\Datasets\IsItCorrect\french_corpus\step_1_reddit.txt'
    with open(write_location, 'w', encoding='utf8') as file:
        file.writelines(list_lines)
    t_write_2 = time.time()
    print("For write_into_file, time is ", t_write_2-t_write_1)


def apple():
    t_apple_1 = time.time()
    lines = load_textfile()
    list_lines = form_re(lines)
    list_lines = clean_punctuations(list_lines)
    list_lines = clean_additional(list_lines)
    write_into_file(list_lines)
    t_apple_2 = time.time()
    print("For apple, time is ", t_apple_2-t_apple_1)


if __name__ == '__main__':
    cProfile.run('apple()', r'D:\Datasets\IsItCorrect\french_corpus\step_1_reddit_pstats.txt')