from tabulate import tabulate
import re
from construct import min_length_qualify, decide_intensity, Sequence
from pathlib import Path
import pickle


def labelify(correct, changed):
    """
    This method is used to make pre-labels (in form of list) and return these (two)
    list (which are of the same len correspoinding to their strings),
    one corressponding to `correct` and other for `changed`.
    :param correct: string which doesn't contain any <START>/<END> tag, denoting correct sentence
    :param changed: string which doesn't contain any <START>/<END> tag, denoting the sentene has changed
    :return: `label_correct`, `label_changed`: list which can be passed to `Type1Label` as `label` variable
    which would be finally converted into label

    `correct` and `changed` might not be of same length (when Grammer absence of words takes place)

    Here are some examples that are to be considered for labeling-dilemma:

    `correct` : We used to play together.
                1   1   1   1   1

    `changed`: We to used play together (sequence)
                1  -1  -1  1    1

    `changed`: We use to together play. (wrong grammar of any type)
               1   -1  1     -1    -1

    `changed`: We car used to play together. (extra words)
               1   -1  1  1  1      1

    `changed`: We used play together. (absent words)
                1  1    -1   1

    Things get difficult when there is a combination of two changes, for example, sequence and absent words:
    `changed` : We to used together. (sequence + absent)
                1  -1  -1   -1
    For now double mistakes (absent words/extra words + sequence) are omitted because labeling becomes immensely
    difficult because the combination of these mistakes can also lead to a correct (albeit different) sentence.
    Eg: `changed` : We play together. (two consecutive words missing may lead to correct sentences)
                    1   -1   1

 Here are some examples of what works for self reference in future:
    `correct`: We used to play together.
               1  1    1   1     1
    `changed`: We car used to play so together.
                1  -1   1    1   1   -1   1
    `changed`: We to used play together.
               1  -1  -1    1      1
    `changed`: We use to together play.
                1   -1  1   -1      -1
    `changed`: We used play together.
                1   1    -1    1
 Examples of what doesn't work:
    `changed`: used play together.
                 -1    -1   0
    `changed`: We to used together.
                1  -1  -1   1
    """

    correct = correct.split(" ")
    changed = changed.split(" ")
    len_cor = len(correct)
    len_cha = len(changed)
    correct_label = [1] * len_cor
    changed_label = [0] * len_cha
    if len_cor == len_cha:
        # No words removed or added, trivial case
        for i in range(len_cor):
            if changed[i] != correct[i]:
                changed_label[i] = -1
            else:
                changed_label[i] = 1

    elif len_cor > len_cha:
        # Words have been removed, so at least one less length
        # The logic here assumes that two words are not skipped in row, need to adapt to overcome this assumption
        skip_flag = False  # in `i`th iteration, if correct[i+1] == changed[i] don't check for `i+1`th iteration
        skip_times = 0  # to keep the track of how many times skipped, so correct[i] = changed[i + skip_times]
        for i in range(len_cor):
            if skip_flag:
                skip_flag = False
                continue
            if i <= len_cha:  # i not more than length of `changed`, still incomplete, doesn't satisfy current test case
                if correct[i] == changed[i - skip_times]:
                    changed_label[i - skip_times] = 1
                elif correct[i+1] == changed[i - skip_times] or correct[i-1] == changed[i - skip_times]:
                    changed_label[i - skip_times] = -1
                    skip_flag = True
                    skip_times += 1
                else:
                    changed_label[i - skip_times] = -1
            else:
                pass
    else:
        # Words have been added
        skip_times = 0  # So we could avoid IndexError in correct due to it's smaller size
        for i in range(len_cha):
            if changed[i] == correct[i - skip_times]:
                changed_label[i] = 1
            else:
                changed_label[i] = -1
                skip_times += 1

    return correct_label, changed_label


class Label:
    """ Old Label to store the sentence as a valid label to be fed into loader """
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


class Type1Label:
    """ New Label which is for Labeling Type1 as well as Type2
    (Type1 and Type2 can be found in `construct.txt`)"""
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.lab_len = len(label)
        print("Value of label_length is ", self.lab_len)

        self.label = label
        # self.label = make_label(label)

    def __str__(self):
        label = [self.label[i] for i in range(self.lab_len)]
        return "{" + "label:{}, ".format(label) + \
               "sentence:{} ".format(self.sentence) + "}"

    def __repr__(self):
        return str(self.__str__)

    def pretty_print(self):
        """
        prints in a pretty format, eg:
            We    used    to    play    together.
            1       1     1       1            1
            We    use    to    together    play.
            1     -1     1          -1       -1
        :return: None
        """
        label = [[self.label[i] for i in range(self.lab_len)]]
        headers = [i for i in self.sentence.split(" ")]
        print(tabulate(label, headers, tablefmt='plain'))

    def store_dict(self):
        """
        This method is same as the method from the old labeling class, which is responsible to make a dict which
        can futher be pickled.
        :return: a `dict` of the following format:
        { "label" : List/Tuple of fixed size 512
          "sentence" : String of variable length
        }
        """

        return {"label": self.label, "sentence": self.sentence}


def make_label(label):
    """
    Used to make label complete of 512 length.
    :param label: list of any length
    :return: list of length 512 with 0s appended to `label`
    """
    for _ in range(512 - len(label)):
        label.append(0)
    return label


# examples to use the class `Type1Label`:
# sen_cor = 'We used to play together.'
# sen_wro = 'We use to together play.'
# lab_cor = [1, 1, 1, 1, 1]
# lab_wro = [1, -1, 1, -1, -1]
#
# label_cor = Type1Label(sentence=sen_cor, label=lab_cor)
# print(label_cor)
# print(len(label_cor.label))
#
# label_wro = Type1Label(sentence=sen_wro, label=lab_wro)
# print(label_wro)
# print(len(label_wro.label))
# label_cor.pretty_print()
# label_wro.pretty_print()


# The following are some try-run cases
testing_file_location = Path("sample_text.txt")
pattern = re.compile("<START>.*?<END>")
db_correct = {}
db_wrong = {}
with open(str(testing_file_location), 'r', encoding='utf-8') as file:
    lines = file.readlines()
    lines = pattern.findall(lines[0])
    total_lines = len(lines)
    print("Total Number of lines are ", total_lines)
    for i, line in enumerate(lines):
        # Here Randomize between `Sequence` and `Grammar` when `Grammar` is defined
        sample = Sequence(line)
        if min_length_qualify(sample.correct[7:-5]):
            decided = decide_intensity()
            print("Intensity Level is ", decided)
            if decided == 0:
                sample.intensity_0()
            elif decided == 1:
                sample.intensity_1()
            elif decided == 2:
                sample.intensity_2()
            elif decided == 3:
                sample.intensity_3()
            else:
                sample.intensity_4()

            correct, changed = sample.correct, sample.changed
            correct_label, changed_label = labelify(correct, changed)

            a = Type1Label(correct, correct_label)
            # a.pretty_print()
            b = Type1Label(changed, changed_label)
            # b.pretty_print()

            db_correct[i] = a.store_dict()
            db_wrong[i] = b.store_dict()

    with open('sample_correct_diff.pkl', 'wb') as wfile:
        pickle.dump(db_correct, wfile)
    with open('sample_wrong_diff.pkl', 'wb') as wfile:
        pickle.dump(db_wrong, wfile)

"""
There is a difference of (47-10)kb = 37 kb to store approx. 30 sentences.
This difference is due to having redundant 0s in the label. Need to decide this storage vs computation trade-off.
Adding both type of pickle files, storing the correct and changed sentences into different pickle files.
"""

# cor = "together."
# cha = "We used to play together."
# cor_label, cha_label = labelify(cor, cha)
# a = Type1Label(cor, cor_label)
# b = Type1Label(cha, cha_label)
# a.pretty_print()
# b.pretty_print()

