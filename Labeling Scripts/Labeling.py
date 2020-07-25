from contextlib import closing
import time
from pathlib import Path
import re
import random
from Label import Label
import copy


def swap_apart(line, k, int_range):
    choices = random.choices(int_range, k=k)
    for i in choices:
        if random.randint(0, 1):
            line[i + 2], line[i] = line[i], line[i + 2]
        else:
            line[i - 2], line[i] = line[i], line[i - 2]
    return line


def swap_adjacent(line, k, int_range):
    choices = random.choices(int_range, k=k)
    for i in choices:
        if random.randint(0, 1):  # if 1 then forward change else backward
            line[i + 1], line[i] = line[i], line[i + 1]
        else:
            line[i - 1], line[i] = line[i], line[i - 1]
    return line


def change_or_no(lines):
    t_change_start = time.time()
    change_lines = []
    random.shuffle(lines)

    for line in lines[:len(lines) // 2]:
        change_lines.append((1, line))
    for line in lines[len(lines) // 2:]:
        change_lines.append((0, line))

    t_change_end = time.time()
    print("Time taken for change_or_no ", t_change_end - t_change_start)
    return change_lines


def change_intensity(lines):
    t_change_i_start = time.time()
    dict_choice = {
        'Intensity {}'.format(i): 0 for i in range(5)
    }
    for i in range(len(lines)):
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


def intensity_0(line):
    if line[0]:  # If change value is 1
        len_sen = len(line[1].split(" "))
        correct_line = line[1].split(" ")
        line = copy.copy(correct_line)
        int_range = [i for i in range(2, len_sen - 2)]
        if random.randint(0, 1):  # choose adjacent if 1 or one word apart
            if len_sen > 20:
                changed = swap_adjacent(line, 5, int_range)
            elif 20 >= len_sen >= 10:
                changed = swap_adjacent(line, 3, int_range)
            elif 10 > len_sen >= 5:
                changed = swap_adjacent(line, 2, int_range)
            else:
                changed = swap_adjacent(line, 1, int_range)
        else:
            int_range[0] += 1
            int_range[-1] -= 1
            if len_sen > 30:
                changed = swap_apart(line, 3, int_range)
            elif 30 >= len_sen > 15:
                changed = swap_apart(line, 2, int_range)
            else:
                changed = swap_apart(line, 1, int_range)
        label = Label(False)
        label.assign(correct_line, changed)
        return label


def intensity_1(line):
    if line[0]:
        print(line[1])
    return line


def intensity_2(line):
    if line[0]:
        print(line[1])
    return line


def intensity_3(line):
    if line[0]:
        print(line[1])
    return line


def intensity_4(line):
    if line[0]:
        print(line[1])
    return line


if __name__ == '__main__':
    pattern = re.compile('<START>.*?<END>')
    with closing(open(Path(r'D:/Datasets/IsItCorrect/testing.txt'), 'r', encoding='utf8')) as file:
        lines = file.readlines()
        lines = pattern.findall(lines[0])
        print("Total number of lines are, ", len(lines))
        len_dict = {}
        for line in lines:
            try:
                len_dict[len(line.split(" "))] += 1
            except KeyError:
                len_dict[len(line.split(" "))] = 1
    lines = change_or_no(lines)
    change_intensity(lines[:len(lines) // 2])
