"""Refer to Labeling-Technique-Beta.txt for more information"""
from contextlib import closing
from pathlib import Path
import re
import random
import pickle


class Label:

    def __init__(self, label):
        self.label = label
        self.sentence = None

    def assign(self, sentence):
        self.sentence = sentence

    def __str__(self):
        return "{" + "label:{}, ".format(self.label) + \
               "sentence:{} ".format(self.sentence) + "}"

    def __repr__(self):
        return str("{" + "label:{}, ".format(self.label) +
                   "sentence:{} ".format(self.sentence) + "}")

    def getList(self):
        return self.sentence.split(" ")

    def store_dict(self):
        label = 1 if self.label else 0
        dictionary = {
            "label": label,
            "sentence": self.sentence
        }
        return dictionary


def swap_apart(line, k, int_range):     # for intensity_0
    choices = random.choices(int_range, k=k)
    for i in choices:
        if random.randint(0, 1):
            line[i + 2], line[i] = line[i], line[i + 2]
        else:
            line[i - 2], line[i] = line[i], line[i - 2]
    return " ".join(line)


def swap_adjacent(line, k, int_range):  # for intensity_0
    choices = random.choices(int_range, k=k)
    for i in choices:
        if random.randint(0, 1):  # if 1 then forward change else backward
            line[i + 1], line[i] = line[i], line[i + 1]
        else:
            line[i - 1], line[i] = line[i], line[i - 1]
    return " ".join(line)


def three_words_swap(line, times):  # for intensity_1
    int_range = [i for i in range(len(line))]
    for time in range(times):
        choices = random.choices(int_range, k=3)
        line[choices[0]], line[choices[1]], line[choices[2]] = line[choices[2]], line[choices[0]], line[choices[1]]
    return line


def change_intensity(lines):
    for i in range(len(lines)):
        if MinLengthQualify(lines[i][7:-5]):
            choice = random.choices([0, 1, 2, 3, 4], weights=[37, 28, 18, 9, 8], k=1)[0]
            if choice == 1:
                lines[i] = intensity_1(lines[i])
            elif choice == 2:
                lines[i] = intensity_2(lines[i])
            elif choice == 3:
                lines[i] = intensity_3(lines[i])
            elif choice == 4:
                lines[i] = intensity_4(lines[i])
            else:
                lines[i] = intensity_0(lines[i])
        else:
            pass
    return lines


def MinLengthQualify(line):
    return len(line.split(" ")) > 5


def intensity_0(line):
    line = line[7:-5].split(" ")
    len_sen = len(line)
    int_range = [i for i in range(1, len_sen - 2)]
    if random.randint(0, 1):    # choose adjacent if 1
        if len_sen > 20:
            line = swap_adjacent(line, 5, int_range)
        elif 20 >= len_sen >= 10:
            line = swap_adjacent(line, 3, int_range)
        elif 10 > len_sen >= 5:
            line = swap_adjacent(line, 2, int_range)
        else:
            line = swap_adjacent(line, 1, int_range)
    else:
        int_range[0] += 1
        int_range[-1] -= 1
        if len_sen > 30:
            line = swap_apart(line, 3, int_range)
        elif 30 >= len_sen > 15:
            line = swap_apart(line, 2, int_range)
        else:
            line = swap_apart(line, 1, int_range)
    return line


def intensity_1(line):
    line = line[7:-5].split(" ")
    len_sen = len(line)
    if len_sen > 40:
        changed = three_words_swap(line[:len_sen // 2], 2)
        changed.append(three_words_swap(line[len_sen // 2:], 2))
    elif 40 >= len_sen > 20:
        changed = three_words_swap(line[:len_sen // 3], 1)
        changed.append(three_words_swap(line[len_sen // 3:2 * len_sen // 3], 1))
        changed.append(three_words_swap(line[2 * len_sen // 3:], 1))
    elif 20 >= len_sen > 10:
        changed = three_words_swap(line[:len_sen // 2], 1)
        changed.append(three_words_swap(line[len_sen // 2:], 1))
    else:
        changed = three_words_swap(line, 1)
    answer = ""
    for value in changed:
        try:
            answer = answer + " " + value
        except:
            answer = answer + " " + " ".join(value)
    return answer


def intensity_2(line):
    line = line[7:-5].split(" ")
    len_sen = len(line)
    init_range = [i for i in range(len_sen)]

    num_words_random = round(len_sen * 0.25)
    choices = random.choices(init_range, k=num_words_random)
    for i in range(len(choices) // 2):
        line[choices[-(i + 1)]], line[choices[i]] = line[choices[i]], line[choices[-(i + 1)]]

    return " ".join(line)


def intensity_3(line):
    line = line[7:-5].split(" ")
    len_sen = len(line)
    init_range = [i for i in range(len_sen)]

    num_words_random = round(len_sen * 0.50)
    choices = random.choices(init_range, k=num_words_random)
    for i in range(len(choices) // 2):
        line[choices[-(i + 1)]], line[choices[i]] = line[choices[i]], line[choices[-(i + 1)]]

    return " ".join(line)


def intensity_4(line):
    line = line[7:-5].split(" ")
    len_sen = len(line)
    init_range = [i for i in range(len_sen)]

    num_words_random = round(len_sen * 0.75)
    choices = random.choices(init_range, k=num_words_random)
    for i in range(len(choices) // 2):
        line[choices[-(i + 1)]], line[choices[i]] = line[choices[i]], line[choices[-(i + 1)]]

    return " ".join(line)


def store_label0(lines):
    """To store the randomized sentence into the pickle file"""
    lines = change_intensity(lines)
    label_list = []
    for line in lines:
        if MinLengthQualify(line):
            label = Label(0)
            label.assign(line)
            label_list.append(label)
        else:
            pass
    return label_list


def store_label1(lines):
    """To store the correct sentence into the pickle file"""
    label_list = []
    for line in lines:
        line = line[7:-5]
        if MinLengthQualify(line):
            label = Label(1)
            label.assign(line)
            label_list.append(label)
    return label_list


if __name__ == '__main__':
    pattern = re.compile("<START>.*?<END>")
    with closing(open(r"D:\Datasets\IsItCorrect\testing-new.txt", 'r', encoding='utf-8')) as file:
        lines = file.readlines()
        lines = pattern.findall(lines[0])
        print("Total number of lines, ", len(lines))
        label_1 = store_label1(lines)
        label_0 = store_label0(lines)
        lines = None

        # print("Label_0 length", len(label_0))
        # print("Label_1 length", len(label_1))
        # print("----------------------------")
        # print("Here's some examples from each:")
        # for i in range(15):
        #     print("label_0 {}:".format(i), label_0[i])
        #     print("label_1 {}:".format(i), label_1[i])
        db_train = {}
        with closing(open(Path(r'D:/Datasets/IsItCorrect/beta_sample_train.pkl'),
                          'ab')) as file:
            for i, line in enumerate(label_0 + label_1):
                db_train[i] = line.store_dict()
            pickle.dump(db_train, file)



