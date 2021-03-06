from contextlib import closing
import time
from pathlib import Path
import re
import random
from Label import Label
import copy
import pickle


def swap_apart(line, k, int_range):     # for intensity_0
    choices = random.choices(int_range, k=k)
    for i in choices:
        if random.randint(0, 1):
            line[i + 2], line[i] = line[i], line[i + 2]
        else:
            line[i - 2], line[i] = line[i], line[i - 2]
    return line


def swap_adjacent(line, k, int_range):  # for intensity_0
    choices = random.choices(int_range, k=k)
    for i in choices:
        if random.randint(0, 1):  # if 1 then forward change else backward
            line[i + 1], line[i] = line[i], line[i + 1]
        else:
            line[i - 1], line[i] = line[i], line[i - 1]
    return line


def three_words_swap(line, times):  # for intensity_1
    int_range = [i for i in range(len(line))]
    for time in range(times):
        choices = random.choices(int_range, k=3)
        line[choices[0]], line[choices[1]], line[choices[2]] = line[choices[2]], line[choices[0]], line[choices[1]]
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
    for i in range(len(lines)):
        if MinLengthQualify(lines[i]):
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
    t_change_i_end = time.time()
    print("time taken for change_intensity ", t_change_i_end - t_change_i_start)
    return lines


def MinLengthQualify(correct_line):
    return len(correct_line[1].split(" ")) > 5


def intensity_0(line):
    if line[0]:  # If change value is 1
        correct_line = line[1].split(" ")
        len_sen = len(correct_line)
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
                print(line)
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
        label.assign(" ".join(correct_line), " ".join(changed))
        return label


def intensity_1(line):
    if line[0]:
        correct_line = line[1].split(" ")[1:-1]  # remove <START> and <END>
        len_sen = len(correct_line)
        line = copy.copy(correct_line)
        correct_line.insert(0, '<START>')
        correct_line.append('<END>')
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
        changed.insert(0, '<START>')
        changed.append('<END>')
        answer = ""
        for value in changed:
            try:
                answer = answer + " " + value
            except:
                answer = answer + " " + " ".join(value)
        label = Label(False)
        label.assign(" ".join(correct_line), answer)
        return label


def intensity_2(line):
    if line[0]:
        correct_line = line[1].split(" ")[1:-1]
        len_sen = len(correct_line)
        line = copy.copy(correct_line)
        init_range = [i for i in range(len_sen)]
        correct_line.insert(0, '<START>')
        correct_line.append('<END>')
        num_words_random = round(len_sen * 0.25)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            line[choices[-(i + 1)]], line[choices[i]] = line[choices[i]], line[choices[-(i + 1)]]
        line.insert(0, '<START>')
        line.append('<END>')
        label = Label(False)
        label.assign(" ".join(correct_line), " ".join(line))
        return label


def intensity_3(line):
    if line[0]:
        correct_line = line[1].split(" ")[1:-1]
        len_sen = len(correct_line)
        line = copy.copy(correct_line)
        init_range = [i for i in range(len_sen)]
        correct_line.insert(0, '<START>')
        correct_line.append('<END>')
        num_words_random = round(len_sen * 0.50)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            line[choices[-(i + 1)]], line[choices[i]] = line[choices[i]], line[choices[-(i + 1)]]
        line.insert(0, '<START>')
        line.append('<END>')
        label = Label(False)
        label.assign(" ".join(correct_line), " ".join(line))
        return label


def intensity_4(line):
    if line[0]:
        correct_line = line[1].split(" ")[1:-1]
        len_sen = len(correct_line)
        line = copy.copy(correct_line)
        init_range = [i for i in range(len_sen)]
        correct_line.insert(0, '<START>')
        correct_line.append('<END>')
        num_words_random = round(len_sen * 0.75)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            line[choices[-(i + 1)]], line[choices[i]] = line[choices[i]], line[choices[-(i + 1)]]
        line.insert(0, '<START>')
        line.append('<END>')
        label = Label(False)
        label.assign(" ".join(correct_line), " ".join(line))
        return label


def convert_to_label(lines):
    label_list = []
    for line in lines:
        if not line[0]:
            label = Label(True)
            label.assign(line[1])
            label_list.append(label)
    return label_list


if __name__ == '__main__':
    pattern = re.compile('<START>.*?<END>')
    with closing(open(Path(r'D:/Datasets/IsItCorrect/testing.txt'), 'r', encoding='utf8')) as file:
        lines = file.readlines()
        lines = pattern.findall(lines[0])
        print("Total number of lines are, ", len(lines))

    lines = change_or_no(lines)
    print("Length of lines ",len(lines))
    lines[:len(lines) // 2] = change_intensity(lines[:len(lines) // 2])
    print("Length of part_1", len(lines))
    lines[len(lines)//2:] = convert_to_label(lines[len(lines)//2:])
    print("Length of part_2", len(lines))
    db_train = {}
    remove_index = 0
    with closing(open(Path(r'D:/Datasets/IsItCorrect/sample_train.pkl'),
                      'ab')) as file:
        for i, line in enumerate(lines[::2]):
            if type(line) in [tuple, str]:
                remove_index += 1
                pass      # Minimum length must have been less than the condition
            else:
                db_train[i-remove_index] = line.store_dict()
        pickle.dump(db_train, file)

    db_train = {}       # clean some memory

    db_test = {}
    remove_index = 0
    with closing(open(Path(r'D:/Datasets/IsItCorrect/sample_test.pkl'),
                      'ab')) as file:
        for i, line in enumerate(lines[1::2]):
            if type(line) in [tuple, str]:
                remove_index += 1
                pass    # Minimum length must have been less than the condition
            else:
                db_test[i - remove_index] = line.store_dict()
        pickle.dump(db_test, file)
