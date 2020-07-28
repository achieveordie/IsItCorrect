"""Refer to Labeling-Technique.txt for more information"""


class Label:

    def __init__(self, correct):
        self.correct = correct
        self.given = None
        self.actual = None

    def assign(self, sentence1, sentence2=None):
        self.actual = sentence1
        if self.correct:
            self.given = sentence1
        else:
            self.given = sentence2

    def __str__(self):
        return "{" + "correct:{}, ".format(self.correct) + \
               "given:{}, ".format(self.given) + \
               "actual:{}".format(self.actual) + "}"

    def __repr__(self):
        return str("{" + "correct:{}, ".format(self.correct) +
                   "given:{}, ".format(self.given) +
                   "actual:{}".format(self.actual) + "}")

    def getList(self):
        return self.actual.split(" "), self.given.split(" ")

    def difference(self):
        if self.correct:
            print("No Difference")
        else:
            a, g = self.getList()
            for i in range(len(a)):
                if a[i] != g[i]:
                    print("Difference at ", i+1, "\n 'In actual': ", a[i], ", 'In given': ", g[i])