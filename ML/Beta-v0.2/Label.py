from tabulate import tabulate


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
