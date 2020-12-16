import numpy as np


class Rouge(object):
    def __init__(self, beta=1.0):

        self.beta = beta
        self.inst_scores = []

    def lcs(self, string, sub):
        if len(string) < len(sub):
            sub, string = string, sub
        lengths = np.zeros((len(string) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(string)][len(sub)]

    def add_inst(self, cand, ref):

        basic_lcs = self.lcs(cand, ref)
        p_denom = len(cand)
        r_denom = len(ref)
        prec = basic_lcs / (p_denom if p_denom > 0. else 0.)
        recall = basic_lcs / (r_denom if r_denom > 0. else 0.)

        if prec != 0 and recall != 0:
            score = ((1 + self.beta ** 2) * prec * recall) / \
                    float(recall + self.beta ** 2 * prec)
        else:
            score = 0.0
        self.inst_scores.append(score)

    def score(self):
        return 1. * sum(self.inst_scores) / len(self.inst_scores)
