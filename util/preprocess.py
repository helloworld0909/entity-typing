from collections import defaultdict
import json
import numpy as np


def getCharSet():
    charStr = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
    return ['PADDING', 'UNKNOWN'] + list(charStr)

def getChar2idx():
    charSet = getCharSet()
    return {v:k for k,v in enumerate(charSet)}

def tokenFrequency(filePathList):
    tokenFreq = defaultdict(int)

    for filePath in filePathList:
        with open(filePath, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                tokens = json.loads(line)['tokens']
                for token in tokens:
                    tokenFreq[token] += 1
    return tokenFreq

def loadWordEmbedding(filePath, dim=100):
    word2vector = {'PADDING': np.zeros(dim), 'UNKNOWN': np.random.uniform(-0.25, 0.25, 100)}
    with open(filePath, 'r', encoding='utf-8') as embeddingFile:
        for line in embeddingFile:
            data_tuple = line.rstrip().split(' ')
            token = data_tuple[0]
            vector = data_tuple[1:]
            word2vector[token] = vector
    return word2vector

def tokenLengthDistribution(token2idx):
    distribution = defaultdict(int)
    for token in token2idx.keys():
        tokenLength = len(token)
        distribution[tokenLength] += 1
    return distribution

def selectPaddingLength(lengthDistribution, ratio=0.99):
    totalCount = sum(lengthDistribution.values())
    threshold = int(totalCount * ratio)
    countSum = 0
    selectedLength = 0
    for length, count in sorted(lengthDistribution.items(), key=lambda kv: kv[0]):
        countSum += count
        selectedLength = length
        if countSum >= threshold:
            break
    return selectedLength

