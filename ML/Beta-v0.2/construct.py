import json
from pathlib import Path
import random
from abc import ABC, abstractmethod
import logging

random.seed(0)

# Opening some files for Grammar:
base_save_location = Path.cwd() / 'Grammar' / 'data'
# base_save_location = Path("D:/Datasets/IsItCorrect/temp")

pos_json_loc = base_save_location / 'pos.json'
pos_json = json.load(open(str(pos_json_loc), 'r', encoding='utf-8'))

word_json_loc = base_save_location / 'word.json'
word_json = json.load(open(str(word_json_loc), 'r', encoding='utf-8'))

lemma_json_loc = base_save_location / 'lemma.json'
lemma_json = json.load(open(str(lemma_json_loc), 'r', encoding='utf-8'))

word_lemma_json_loc = base_save_location / 'word_lemma.json'
word_lemma_json = json.load(open(str(word_lemma_json_loc), 'r', encoding='utf-8'))

constant_grammar = ('ABR', 'NAM', 'PUN', 'PUN:cit', 'SENT', 'SYM')
# inter_pos = ('PRO', 'VER')
pronoun_grammar = [pos for pos in pos_json.keys() if pos[0:3] == "PRO"]
verb_grammar = [pos for pos in pos_json.keys() if pos[0:3] == "VER"]


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
        self.correct = line[7:-5].strip()
        self.changed = line[7:-5].strip()
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
        if self.choice is not None:  # need to ensure we don't call this method more than once
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
            return " ".join(line).lstrip()

        def swap_adjacent(line, k, int_range):
            choices = random.choices(int_range, k=k)
            for i in choices:
                if random.randint(0, 1):  # if 1 then forward change else backward
                    line[i + 1], line[i] = line[i], line[i + 1]
                else:
                    line[i - 1], line[i] = line[i], line[i - 1]
            return " ".join(line).lstrip()

        # self.changed = self.correct.split(" ")
        self.changed = self.changed.split(" ")
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

        # self.changed = self.correct.split(" ")
        self.changed = self.changed.split(" ")
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

        self.changed = answer.lstrip()

    def intensity_2(self):
        # self.changed = self.correct.split(" ")
        self.changed = self.changed.split(" ")
        len_sen = len(self.changed)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.25)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.changed[choices[-(i + 1)]], self.changed[choices[i]] = self.changed[choices[i]], self.changed[choices[-(i + 1)]]

        self.changed = " ".join(self.changed).lstrip()

    def intensity_3(self):
        # self.changed = self.correct.split(" ")
        self.changed = self.changed.split(" ")
        len_sen = len(self.changed)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.50)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.changed[choices[-(i + 1)]], self.changed[choices[i]] = self.changed[choices[i]], self.changed[choices[-(i + 1)]]

        self.changed = " ".join(self.changed).lstrip()

    def intensity_4(self):
        # self.changed = self.correct.split(" ")
        self.changed = self.changed.split(" ")
        len_sen = len(self.changed)
        init_range = [i for i in range(len_sen)]

        num_words_random = round(len_sen * 0.75)
        choices = random.choices(init_range, k=num_words_random)
        for i in range(len(choices) // 2):
            self.changed[choices[-(i + 1)]], self.changed[choices[i]] = self.changed[choices[i]], self.changed[choices[-(i + 1)]]

        self.changed = " ".join(self.changed).lstrip()


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
    to have an equal (or appropriate) amount of mistakes in the data for the model.

    This is a subclass to `Changes` and has all the restrictions that `Sequence` had.
    """
    def __init__(self, line):
        self.correct = line[7:-5]
        self.changed = line[7:-5].split(" ")
        self.length = len(self.changed)
        self.choice = None
        self.words = {}
        self.unchangeable = False

    def make_choice(self):
        super(Grammar, self).make_choice()
        self.choice = random.choices([0, 1, 2, 3, 4], weights=[37, 28, 18, 9, 8], k=1)[0]

    def make_change(self):
        super(Grammar, self).make_change()
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

    def choose_word(self, percent_of_words, end_range, start_range=0):
        """
        Choose `percent_of_words` words at random from `self.changed` and
        store it in self.words as {index:word}
        :param end_range: what's the last index(wrt the length) to sample from?
        :param start_range: what's the first index to sample from? default to 0.
        :param percent_of_words: <float/int> if float then between 0.0 and 1.0, denotes percent of words to select.
                                 If int then denotes number of words to select.
        :return: None, changes are made in self.words
        """
        assert start_range < end_range <= self.length, "range limit broken."
        if type(percent_of_words) == float:
            assert 0.0 < percent_of_words < 1.0, "float value should be between 0 and 1 "
            number_of_words = int(self.length * percent_of_words)
            indices = random.sample(range(start_range, end_range), k=number_of_words)
            for i in indices:
                self.words[i] = self.changed[i]
        elif type(percent_of_words) == int:
            assert self.length > percent_of_words, "percent_of_words are more than length in the sentence"
            assert percent_of_words > 0, "A negative percent_of_words value encountered."
            indices = random.sample(range(start_range, end_range), k=percent_of_words)
            for i in indices:
                self.words[i] = self.changed[i]
        else:
            raise (ValueError, "percent_of_words should either be an int or float got {}".format(type(percent_of_words)))

    def remove_words_if_necessary(self):
        """
        There is a global `constant_grammar` list. These are the POSs that are not to be altered.
        Remove such entries.
        :return: None, changes are being made in `words` attribute (if necessary).
        """
        keys_to_remove = []
        for index, word in self.words.items():
            try:
                if word_json[word] in constant_grammar:
                    keys_to_remove.append(index)
            except KeyError:
                logging.info(word)

        for key in keys_to_remove:
            del self.words[key]

    def change_words_from_POS(self, single_pair=None):
        """
        After we have accumulated words that are to be changed, change them using their POS.
        If POS belong either to either PRO or VERB, then change within POS.
        :param single_pair: If this is not None that means we only want to alter single word and single pair
                            will be a dict of {"index":index, "word":word}, this is the case when lemma of a word
                            doesn't exist so it falls back to using this method.
        :return: None, changes are being made in `changed` attribute.
        """
        if single_pair is not None:
            try:
                pos = word_json[single_pair["word"]]
            except KeyError:
                logging.info(single_pair["word"])
                return None
            if pos[0:3] == 'PRO':
                which_pos = random.randrange(start=0, stop=len(pronoun_grammar))
                self.changed[single_pair["index"]] = random.choice(pos_json[pronoun_grammar[which_pos]])
            elif pos[0:3] == 'VER':
                which_pos = random.randrange(start=0, stop=len(verb_grammar))
                self.changed[single_pair["index"]] = random.choice(pos_json[verb_grammar[which_pos]])
            else:
                self.changed[single_pair["index"]] = random.choice(pos_json[pos])
        else:  # method was invoked after `change_words_from_lemma` failed.
            for index, word in self.words.items():
                try:
                    pos = word_json[word]
                except KeyError:
                    logging.info(word)
                    return None
                if pos[0:3] == 'PRO':
                    which_pos = random.randrange(start=0, stop=len(pronoun_grammar))
                    self.changed[index] = random.choice(pos_json[pronoun_grammar[which_pos]])
                elif pos[0:3] == 'VER':
                    which_pos = random.randrange(start=0, stop=len(pronoun_grammar))
                    self.changed[index] = random.choice(pos_json[pronoun_grammar[which_pos]])
                else:
                    self.changed[index] = random.choice(pos_json[word_json[word]])

    def change_words_from_lemma(self):
        """
        Read change_words_from_POS, but for lemmas.
        :return: None, changes are being made in `changed` attribute.
        """
        for index, word in self.words.items():
            try:
                self.changed[index] = random.choice(lemma_json[word_lemma_json[word]])
            except KeyError:
                self.change_words_from_POS(single_pair={"index": index, "word": word})

    def do_changes(self):
        """
        Simple patching of other methods that is to be used by every intensity level after `words`
        attribute is filled as per the conditions.
        :return: None, final changes are made in `changed` attribute. Essentially converting it back to a string.
        """
        self.remove_words_if_necessary()
        if self.words is None:
            self.unchangeable = True

        if not self.unchangeable:
            choice = random.choices([0, 1, 2], weights=[20, 30, 50], k=1)[0]  # 0 for pos, 1 for lemma, 2 for both.
            if choice == 0:
                self.change_words_from_POS()
            elif choice == 1:
                self.change_words_from_lemma()
            else:
                self.change_words_from_lemma()
                self.change_words_from_POS()

    def intensity_0(self):
        if self.length >= 40:
            self.choose_word(percent_of_words=2, end_range=self.length//2)
            self.choose_word(percent_of_words=2, start_range=self.length//2, end_range=self.length)
        elif 40 > self.length > 20:
            self.choose_word(percent_of_words=1, end_range=self.length//3)
            self.choose_word(percent_of_words=1, start_range=self.length//3, end_range=2*self.length//3)
            self.choose_word(percent_of_words=1, start_range=2*self.length//3, end_range=self.length)
        elif 20 >= self.length > 10:
            self.choose_word(percent_of_words=2, end_range=self.length)
        else:
            self.choose_word(percent_of_words=1, end_range=self.length)

        self.do_changes()
        self.changed = " ".join(self.changed).lstrip()

    def intensity_1(self):
        if self.length >= 40:
            self.choose_word(percent_of_words=4, end_range=self.length//2)
            self.choose_word(percent_of_words=4, start_range=self.length//2, end_range=self.length)
        elif 40 > self.length > 20:
            self.choose_word(percent_of_words=2, end_range=self.length//3)
            self.choose_word(percent_of_words=2, start_range=self.length//3, end_range=2*self.length//3)
            self.choose_word(percent_of_words=2, start_range=2*self.length//3, end_range=self.length)
        elif 20 >= self.length > 10:
            self.choose_word(percent_of_words=2, end_range=self.length//2)
            self.choose_word(percent_of_words=2, start_range=self.length//2, end_range=self.length)
        else:
            self.choose_word(percent_of_words=2, end_range=self.length)

        self.do_changes()
        self.changed = " ".join(self.changed).lstrip()

    def intensity_2(self):
        self.choose_word(percent_of_words=0.1, end_range=self.length)

        self.do_changes()
        self.changed = " ".join(self.changed).lstrip()

    def intensity_3(self):
        self.choose_word(percent_of_words=0.25, end_range=self.length)

        self.do_changes()
        self.changed = " ".join(self.changed).lstrip()

    def intensity_4(self):
        self.choose_word(percent_of_words=0.5, end_range=self.length)

        self.do_changes()
        self.changed = " ".join(self.changed).lstrip()
