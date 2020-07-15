"""Refer to Labeling-Technique.txt for more information"""

class Label:

    def __init__(self, correct):
        self.correct = correct
        self.given = None
        self.actual = None

    def assign(self, sentence1, sentence2=None):
        self.given = sentence1
        if self.correct:
            self.actual = sentence1
        else:
            self.actual = sentence2

    def __str__(self):
        return "{" + "correct:{}, ".format(self.correct) + \
               "given:{}, ".format(self.given) + \
               "actual:{}".format(self.actual) + "}"

    def __repr__(self):
        return str("{" + "correct:{}, ".format(self.correct) + \
                   "given:{}, ".format(self.given) + \
                   "actual:{}".format(self.actual) + "}")