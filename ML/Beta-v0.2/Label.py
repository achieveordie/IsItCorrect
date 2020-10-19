from tabulate import tabulate
import re
from construct import min_length_qualify, decide_intensity, Sequence


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
    """



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

        self.label = make_label(label)

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


def make_label(label):
    """
    Used to make label complete of 512 length.
    :param label: list of any length
    :return: numpy array of length 512 with 0s appended to `label`
    """
    for _ in range(512 - len(label)):
        label.append(0)
    return label


# examples:
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


testing_file_location = r"D:\Datasets\IsItCorrect\testing-new.txt"
pattern = re.compile("<START>.*?<END>")
with open(testing_file_location, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    lines = pattern.findall(lines[0])
    print("Total Number of lines are ", len(lines))
    sample = lines[5]
    del lines
print(sample)
sample = Sequence(sample)
if min_length_qualify(sample.correct[7:-5]):
    decided = decide_intensity()
    print("Intensity Level is ", decided)

