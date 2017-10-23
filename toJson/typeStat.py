import json
import math
from itertools import combinations
from collections import defaultdict



class TypeStat(object):

    label2idx = {}
    labelFreq = defaultdict(int)
    labelPairFreq = defaultdict(int)

    def __init__(self, filepath):
        with open(filepath + 'selected types.txt', 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                label, freq = line.strip().split('\t')
                self.label2idx[label] = len(self.label2idx)

        with open(filepath + 'train.json', 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                jsonObj = json.loads(line)
                for mention in jsonObj['mentions']:
                    labels = mention['labels']
                    labelPairs = combinations(labels, 2)
                    for label in labels:
                        self.labelFreq[label] += 1
                    for labelPair in labelPairs:
                        self.labelPairFreq[tuple(sorted(labelPair))] += 1
        self.labelSum = sum(self.labelFreq.values())
        self.labelPairSum = sum(self.labelPairFreq.values())

    def getLabelPairProb(self, labelPair):
        pair = tuple(sorted(labelPair))
        return (self.labelPairFreq.get(pair, 0) + 1) / float(self.labelPairSum + (len(self.label2idx)^2) / 2)

    def getLabelProb(self, label):
        return self.labelFreq.get(label, 0) / float(self.labelSum)

    def getPMI(self, labelPair):
        p_ab = self.getLabelPairProb(labelPair)
        p_a = self.getLabelProb(labelPair[0])
        p_b = self.getLabelProb(labelPair[1])
        return math.log(p_ab / (p_a * p_b))

    @staticmethod
    def isParent(parent, sub):
        if sub.startswith(parent):
            return True
        else:
            return False


if __name__ == '__main__':
    typeStat = TypeStat('E:/python_workspace/entity-typing/data/baike/')
    labels = list(typeStat.label2idx.keys())
    labelPMI = {}
    for pair in combinations(labels, 2):
        labelPMI[pair] = typeStat.getPMI(pair)
    with open('E:/python_workspace/entity-typing/data/baike/typePMI.txt', 'w', encoding='utf-8') as outputFile:
        for pair, value in sorted(labelPMI.items(), key=lambda pv: pv[1]):
            outputFile.write('\t'.join(pair) + '\t' + str(value) + '\n')
