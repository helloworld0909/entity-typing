import logging
import json
import numpy as np
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from util.corpusEN import CorpusEN
from util import preprocess


class MyCorpus(CorpusEN):

    embeddingFilePath = 'data/glove.6B.100d.txt'

    def __init__(self, filePathList):
        super(MyCorpus, self).__init__()
        self.initMappings(filePathList)
        self.initMaxTokenLength()
        self.initTokenIdx2charVector(self.maxTokenLength)
        self.initWordEmbedding(self.embeddingFilePath, dim=100)
        self.initCharEmbedding(30)


    def initMappings(self, filePathList):

        for filePath in filePathList:
            with open(filePath, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    sent = json.loads(line)
                    tokens = sent['tokens']

                    for token in tokens:
                        if token not in self.token2idx:
                            self.token2idx[token] = len(self.token2idx)

                    for mention in sent['mentions']:
                        labels = mention['labels']
                        for label in labels:
                            if label not in self.label2idx:
                                self.label2idx[label] = len(self.label2idx)
        self.vocabSize = len(self.token2idx)
        self.labelDim = len(self.label2idx)
        logging.info('Vocabulary size: ' + str(self.vocabSize))
        logging.info('Label dim: '+ str(self.labelDim))


    def initMaxTokenLength(self):
        tokenLengthDistribution = preprocess.tokenLengthDistribution(self.token2idx)
        self.maxTokenLength = preprocess.selectPaddingLength(tokenLengthDistribution, ratio=0.99)
        logging.info('Max token length: ' + str(self.maxTokenLength))


    def loadFile(self, filePath):
        X_entity = []
        X_left = []
        X_right = []
        y = []

        labelDim = len(self.label2idx)

        with open(filePath, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                body = json.loads(line)
                sentence = body['tokens']
                idxSentence = list(map(lambda token: self.token2idx.get(token, 1), sentence))


                for mention in body['mentions']:
                    start = mention['start']
                    end = mention['end']
                    entity = idxSentence[start:end]
                    leftContext = idxSentence[:start]
                    rightContext = idxSentence[end:]
                    idxLabels = list(map(lambda label: self.label2idx[label], mention['labels']))
                    oneHotVector = self.oneHotEncode(idxLabels, labelDim)

                    X_entity.append(entity)
                    X_left.append(leftContext)
                    X_right.append(rightContext)
                    y.append(oneHotVector)

        logging.debug(X_entity[0])
        logging.debug(X_left[0])
        logging.debug(X_right[0])

        # TODO: dynamically select maxlen
        X_entity = pad_sequences(X_entity, maxlen=6)
        X_left = pad_sequences(X_left, maxlen=130, truncating='pre')
        X_right = pad_sequences(X_right, maxlen=130, truncating='post')
        y = np.asarray(y)

        return [X_left, X_entity, X_right], y


    @staticmethod
    def oneHotEncode(idxLabels, labelDim):
        vector = np.zeros(labelDim, dtype='float32')
        for idx in idxLabels:
            vector[idx] = 1.0
        return vector


    @staticmethod
    def oneHotDecode(y):
        labelDict = defaultdict(set)
        for oneHotVector in y:
            sentIdx = len(labelDict)
            for idx, value in enumerate(oneHotVector):
                if value != 0:
                    labelDict[sentIdx].add(idx)
        return labelDict


    @staticmethod
    def topK(y, topK=2):
        argList = y.argsort()[:, -topK:]
        batch_size = y.shape[0]
        dim = y.shape[1]
        predictions = np.zeros((batch_size, dim))

        for idx, args in enumerate(argList):
            for arg in args:
                predictions[idx, arg] = 1

        return predictions


    @staticmethod
    def threshold(y, threshold=0.5):
        ufunc = lambda prob: int(prob > threshold)
        opt = np.vectorize(ufunc)
        return opt(y)


    @staticmethod
    def hybrid(y, threshold=0.5):
        topArgList = y.argsort()[:, -1]
        batch_size = y.shape[0]
        dim = y.shape[1]
        predictions = np.zeros((batch_size, dim))

        for idx, arg in enumerate(topArgList):
            predictions[idx, arg] = 1

        for sentIdx, probVec in enumerate(y):
            for idx, prob in enumerate(probVec):
                if prob > threshold:
                    predictions[sentIdx, idx] = 1

        return predictions





    # TODO: AFET训练集去噪