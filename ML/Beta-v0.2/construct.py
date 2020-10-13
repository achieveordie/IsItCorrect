from contextlib import closing
from pathlib import Path
import re
import random
import pickle


class Sequence:
    """ Class for all Sequence-based changes, created a wrapper from labeling.py in Beta-v0.1"""
    def __init__(self, line):
        self.line = line

    def intensity_0(self):

        def swap_apart(line, k, int_range):
            choices = random.choices(int_range, k=k)
            for i in choices:
                if random.randint(0, 1):
                    line[i + 2], line[i] = line[i], line[i + 2]
                else:
                    line[i - 2], line[i] = line[i], line[i - 2]
            return " ".join(line)

        def swap_adjacent(line, k, int_range):
            choices = random.choices(int_range, k=k)
            for i in choices:
                if random.randint(0, 1):  # if 1 then forward change else backward
                    line[i + 1], line[i] = line[i], line[i + 1]
                else:
                    line[i - 1], line[i] = line[i], line[i - 1]
            return " ".join(line)

        self.line = self.line[7:-5].split(" ")
        len_sen = len(self.line)
        int_range = [i for i in range(1, len_sen - 2)]
        if random.randint(0, 1):  # choose adjacent if 1
            if len_sen > 20:
                self.line = swap_adjacent(self.line, 5, int_range)
            elif 20 >= len_sen >= 10:
                self.line = swap_adjacent(self.line, 3, int_range)
            elif 10 > len_sen >= 5:
                self.line = swap_adjacent(self.line, 2, int_range)
            else:
                self.line = swap_adjacent(self.line, 1, int_range)
        else:
            int_range[0] += 1
            int_range[-1] -= 1
            if len_sen > 30:
                self.line = swap_apart(self.line, 3, int_range)
            elif 30 >= len_sen > 15:
                self.line = swap_apart(self.line, 2, int_range)
            else:
                self.line = swap_apart(self.line, 1, int_range)
        return self.line

    def intensity_1(self):

        def three_words_swap(line, times):
            int_range = [i for i in range(len(line))]
            for time in range(times):
                choices = random.choices(int_range, k=3)
                line[choices[0]], line[choices[1]], line[choices[2]] = line[choices[2]], line[choices[0]], line[
                    choices[1]]
            return line

        self.line = self.line[7:-5].split(" ")
        len_sen = len(self.line)
        if len_sen > 40:
            changed = three_words_swap(self.line[:len_sen // 2], 2)
            changed.append(three_words_swap(self.line[len_sen // 2:], 2))
        elif 40 >= len_sen > 20:
            changed = three_words_swap(self.line[:len_sen // 3], 1)
            changed.append(three_words_swap(self.line[len_sen // 3:2 * len_sen // 3], 1))
            changed.append(three_words_swap(self.line[2 * len_sen // 3:], 1))
        elif 20 >= len_sen > 10:
            changed = three_words_swap(self.line[:len_sen // 2], 1)
            changed.append(three_words_swap(self.line[len_sen // 2:], 1))
        else:
            changed = three_words_swap(self.line, 1)
        answer = ""
        for value in changed:
            try:
                answer = answer + " " + value
            except:
                answer = answer + " " + " ".join(value)
        return answer

    def intensity_2(self):
        self.line = self.line[7:-5].split(" ")
        len_sen = len(self.line)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.25)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.line[choices[-(i + 1)]], self.line[choices[i]] = self.line[choices[i]], self.line[choices[-(i + 1)]]

        return " ".join(self.line)

    def intensity_3(self):
        self.line = self.line[7:-5].split(" ")
        len_sen = len(self.line)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.50)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.line[choices[-(i + 1)]], self.line[choices[i]] = self.line[choices[i]], self.line[choices[-(i + 1)]]

        return " ".join(self.line)

    def intensity_4(self):
        self.line = self.line[7:-5].split(" ")
        len_sen = len(self.line)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.75)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.line[choices[-(i + 1)]], self.line[choices[i]] = self.line[choices[i]], self.line[choices[-(i + 1)]]

        return " ".join(self.line)


class Grammer:
    """Class for all Grammer-based changes to be made for one sentence, added in B-v0.2"""
    def __init__(self, line):
        self.line = line

    def tense(self):
        """ Changing tense of some part of text"""
        pass

    def gender(self):
        """ Changing gender of objects/people"""
        pass

    def person(self):
        """ Changing pov """
        pass

    def consistent(self):
        """ Changing consistency plural """
        pass

    def wrong_words(self):
        """ Adding wrong words """
        pass

    def absent_words(self):
        """ Remove some words """
        pass

    def spelling_errors(self):
        """ create errors in spellings """
        pass

    def extra_words(self):
        """ Add extra, unnessary words"""
        pass
