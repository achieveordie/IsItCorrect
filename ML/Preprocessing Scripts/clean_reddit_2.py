from contextlib import closing
import cProfile
import time
import multiprocessing as mp
import re


def load_vocab():
    t_vocab_1 = time.time()
    vocab_location = r'vocab.txt'
    with closing(open(vocab_location, 'r', encoding='utf8')) as file:
        vocab = file.readlines()
    vocab = vocab[0].split(" ")
    t_vocab_2 = time.time()
    print("For load_vocab, time is ", t_vocab_2-t_vocab_1)
    return vocab


def load_lines():
    t_lines_1 = time.time()
    reddit_location = r'D:\Datasets\IsItCorrect\french_corpus\step_1_reddit.txt'
    with closing(open(reddit_location, 'r', encoding='utf8')) as file:
        lines = file.readlines()
    pattern = re.compile('<START>.*?<END>')
    list_lines = pattern.findall(lines[0])
    for i in range(len(list_lines)):
        list_lines[i] = list_lines[i].split(" ")
    t_lines_2 = time.time()
    print("For load_lines, time is ", t_lines_2-t_lines_1)
    return list_lines


def divide_lines(num_processes):
    t_divide_1 = time.time()
    list_lines = load_lines()
    len_lines = len(list_lines)
    divided = []
    for i in range(num_processes):
        if i != num_processes-1:
            divided.append(list_lines[i*int(len_lines/num_processes): (i+1)*int(len_lines/num_processes)])
        else:
            divided.append(list_lines[i*int(len_lines/num_processes):])
    t_divide_2 = time.time()
    print("For divide_line, time is ", t_divide_2-t_divide_1)
    return divided


def make_unk(list_lines, process_num, vocab):
    t_makeunk_1 = time.time()
    for i in range(len(list_lines)):
        for j in range(1, len(list_lines[i])-1):
            if list_lines[i][j] not in vocab:
                list_lines[i][j] = '<UNK>'
    t_make_unk_2 = time.time()
    print("For make_unk{}, time is ".format(process_num),
          t_make_unk_2-t_makeunk_1)
    write_unk(list_lines, process_num)


def write_unk(list_lines, process_num):
    t_writeunk_1 = time.time()
    write_location = r'D:\Datasets\IsItCorrect\french_corpus\step_2_reddit{}.txt'.format(process_num)
    for i in range(len(list_lines)):
        list_lines[i] = " ".join(list_lines[i])
    with closing(open(write_location, 'w', encoding="utf8")) as file:
        file.writelines(list_lines)
    t_writeunk_2 = time.time()
    print("For write_unk{}, time is ".format(process_num),
          t_writeunk_2-t_writeunk_1)
    exit()


def apple():
    num_processes = 2
    vocab = load_vocab()
    divided = divide_lines(num_processes)
    processes_list = []
    for _ in range(num_processes):
        processes_list.append(mp.Process(target=make_unk, args=(divided[_], _, vocab)))
    for process in processes_list:
        process.start()


if __name__ == '__main__':
    cProfile.run('apple()', r'D:\IsItCorrect\Preprocessing Scripts\pstats_\cleanstats_2.txt')