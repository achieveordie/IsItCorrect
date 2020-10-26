from contextlib import closing
from pathlib import Path
import re
import random
import pickle
from abc import ABC, abstractmethod


def min_length_qualify(line):
    return len(line.split(" ")) > 5


class Changes(ABC):
    """
    This is a base class to `Sequence` and `Grammar` which contains two abstract methods:
    `make_choice` is responsible for making choice as to which changes are to be made in either class.
    `make_change` is responsible for instantiating the actual implementation, resulting in
    `self.changed` to contain the changed sentence in their respective classes, so one only needs to call
    this method after creating the object.
    """
    @abstractmethod
    def make_choice(self):
        pass

    @abstractmethod
    def make_change(self):
        pass


class Sequence(Changes):
    """
    Class for all Sequence-based changes, created a wrapper from labeling.py in Beta-v0.1
    where each intensity level changes the sequence of words.
    The sentence passed SHOULD be enclosed within <START>...<END> which is removed and
    stored into `self.correct` which doesn't change with time and the changes made are
    reflected only in `self.changed` which is by default None.

    The only method that is intended to be called from the outside is `make_change` which handles all other methods.

    for every `intensity` method:
    Takes `self.correct` and `self.changed` and makes changes to `self.changed` which can be called from
    outside of the class.

    The functionality of each level is documented in `Beta/labeling-Technique-Beta.txt`.
        :return: None
    """
    def __init__(self, line):
        self.correct = line[7:-5]
        self.changed = None
        self.choice = None

    def make_choice(self):
        """
        This method is useful for making choices as to what will be the level of intensity of change,
        the logic of why these weights are chosen is documented in `Beta/Labeling-Technique-Beta.txt`
        :return: None
        """
        super().make_choice()
        self.choice = random.choices([0, 1, 2, 3, 4], weights=[37, 28, 18, 9, 8], k=1)[0]

    def make_change(self):
        """
        This method uses all other methods to make a Sequence change
        :return: None
        """
        super(Sequence, self).make_change()
        if self.choice is not None: # need to ensure we don't call this method more than once
            print("Changes have already been made, call `changed` attribute to get changed sentence")
        else:
            self.make_choice()
            if self.choice == 0:
                self.intensity_0()
            elif self.choice == 1:
                self.intensity_1()
            elif self.choice == 2:
                self.intensity_2()
            elif self.choice == 3:
                self.intensity_3()
            else:
                self.intensity_4()

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

        self.changed = self.correct.split(" ")
        len_sen = len(self.changed)
        int_range = [i for i in range(1, len_sen - 2)]
        if random.randint(0, 1):  # choose adjacent if 1
            if len_sen > 20:
                self.changed = swap_adjacent(self.changed, 5, int_range)
            elif 20 >= len_sen >= 10:
                self.changed = swap_adjacent(self.changed, 3, int_range)
            elif 10 > len_sen >= 5:
                self.changed = swap_adjacent(self.changed, 2, int_range)
            else:
                self.changed = swap_adjacent(self.changed, 1, int_range)
        else:
            int_range[0] += 1
            int_range[-1] -= 1
            if len_sen > 30:
                self.changed = swap_apart(self.changed, 3, int_range)
            elif 30 >= len_sen > 15:
                self.changed = swap_apart(self.changed, 2, int_range)
            else:
                self.changed = swap_apart(self.changed, 1, int_range)

    def intensity_1(self):

        def three_words_swap(line, times):
            int_range = [i for i in range(len(line))]
            for time in range(times):
                choices = random.choices(int_range, k=3)
                line[choices[0]], line[choices[1]], line[choices[2]] = line[choices[2]], line[choices[0]], line[
                    choices[1]]
            return line

        self.changed = self.correct.split(" ")
        len_sen = len(self.changed)
        if len_sen > 40:
            changed = three_words_swap(self.changed[:len_sen // 2], 2)
            changed.append(three_words_swap(self.changed[len_sen // 2:], 2))
        elif 40 >= len_sen > 20:
            changed = three_words_swap(self.changed[:len_sen // 3], 1)
            changed.append(three_words_swap(self.changed[len_sen // 3:2 * len_sen // 3], 1))
            changed.append(three_words_swap(self.changed[2 * len_sen // 3:], 1))
        elif 20 >= len_sen > 10:
            changed = three_words_swap(self.changed[:len_sen // 2], 1)
            changed.append(three_words_swap(self.changed[len_sen // 2:], 1))
        else:
            changed = three_words_swap(self.changed, 1)
        answer = ""
        for value in changed:
            try:
                answer = answer + " " + value
            except:
                answer = answer + " " + " ".join(value)

        self.changed = answer

    def intensity_2(self):
        self.changed = self.correct.split(" ")
        len_sen = len(self.changed)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.25)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.changed[choices[-(i + 1)]], self.changed[choices[i]] = self.changed[choices[i]], self.changed[choices[-(i + 1)]]

        self.changed = " ".join(self.changed)

    def intensity_3(self):
        self.changed = self.correct.split(" ")
        len_sen = len(self.changed)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.50)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.changed[choices[-(i + 1)]], self.changed[choices[i]] = self.changed[choices[i]], self.changed[choices[-(i + 1)]]

        self.changed = " ".join(self.changed)

    def intensity_4(self):
        self.changed = self.correct.split(" ")
        len_sen = len(self.changed)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.75)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.changed[choices[-(i + 1)]], self.changed[choices[i]] = self.changed[choices[i]], self.changed[choices[-(i + 1)]]

        self.changed = " ".join(self.changed)


class Grammar(Changes):
    """
    Class for all Grammar-based changes to be made for one sentence, added in B-v0.2.
    Just like `Sequence` class, this contains methods to change the grammar of a sentence in order
    to emulate the mistakes that a learner would make. This is a prototype class which contains
    only 8 methods (8 different ways) to make a grammatical error, this needs to be further expanded
    in order to include more complex errors.

    Unlike `Sequence` class, where every sentences are equally valid, here we need to ensure that in
    order to make a grammatical error, we need to have that grammatical structure in the given sentence
    and hence the abstractmethod `make_choice()` from base class will be responsible in
    identifying this structure. This might not the case for every error (like spelling errors, absent words) and
    hence a probabilistic framework is also required similar to `Sequence`'s `make_choice()` in order
    to have an equal (or appropiate) amount of mistakes in the data for the model.

    This is a subclass to `Changes` and has all the restrictions that `Sequence` had.
    """
    def __init__(self, line):
        self.correct = line[7:-5]
        self.changed = None
        self.choice = None

    def make_choice(self):
        super(Grammar, self).make_choice()
        pass

    def make_change(self):
        super(Grammar, self).make_change()
        pass

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
