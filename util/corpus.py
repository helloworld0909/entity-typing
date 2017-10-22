import logging
import json
import numpy as np
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from util.corpusEN import CorpusEN
from util.corpusCN import CorpusCN
from util import preprocess


class MyCorpus(CorpusEN):

    embeddingFilePath = 'data/glove.6B.100d.txt'

    def __init__(self, filePathList):
        super(MyCorpus, self).__init__()
        self.initMappings(filePathList, cutoff=10)
        self.initMaxTokenLength()
        self.initTokenIdx2charVector(self.maxTokenLength)
        self.initWordEmbedding(self.embeddingFilePath, dim=100)
        self.initCharEmbedding(30)


    def initMappings(self, filePathList, cutoff=10):

        tokenFreq = preprocess.tokenFrequency(filePathList)
        for filePath in filePathList:
            with open(filePath, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    sent = json.loads(line)
                    tokens = sent['tokens']

                    for token in tokens:
                        if token not in self.token2idx and tokenFreq.get(token, 0) >= cutoff:
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
                    if not idxLabels:
                        continue
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

    def loadFileSingleSent(self, filePath):
        X = []
        X_mentionTag = []
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
                    mentionTag = np.zeros(len(idxSentence), dtype='float32')
                    mentionTag[start:end] = 1
                    idxLabels = list(map(lambda label: self.label2idx[label], mention['labels']))
                    oneHotVector = self.oneHotEncode(idxLabels, labelDim)

                    X.append(idxSentence)
                    X_mentionTag.append(mentionTag)
                    y.append(oneHotVector)

        logging.debug(X[0])
        logging.debug(X_mentionTag[0])
        logging.debug(y[0])

        X = pad_sequences(X, maxlen=260)
        X_mentionTag = pad_sequences(X_mentionTag, maxlen=260)
        X_mentionTag = np.expand_dims(X_mentionTag, axis=-1)
        y = np.asarray(y)

        return [X, X_mentionTag], y

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


class MyCorpusCN(CorpusCN):

    def __init__(self, filePathList):
        super(MyCorpusCN, self).__init__()
        self.initMappings(filePathList, cutoff=10)
        self.initWordEmbedding(dim=100)

    def initMappings(self, filePathList, cutoff=10):
        tokenFreq = preprocess.tokenFrequency(filePathList)
        for filePath in filePathList:
            with open(filePath, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    sent = json.loads(line)
                    tokens = sent['tokens']

                    for token in tokens:
                        if token not in self.token2idx and tokenFreq.get(token, 0) >= cutoff:
                            self.token2idx[token] = len(self.token2idx)

                    for mention in sent['mentions']:
                        labels = mention['labels']
                        for label in labels:
                            if label not in self.label2idx:
                                self.label2idx[label] = len(self.label2idx)
        self.vocabSize = len(self.token2idx)
        self.labelDim = len(self.label2idx)
        logging.info('Vocabulary size: ' + str(self.vocabSize))
        logging.info('Label dim: ' + str(self.labelDim))

    def loadFile(self, filePath):
        from models.lstmCN import left_length, entity_length, right_length

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
                    if not idxLabels:
                        continue
                    oneHotVector = self.oneHotEncode(idxLabels, labelDim)

                    X_entity.append(entity)
                    X_left.append(leftContext)
                    X_right.append(rightContext)
                    y.append(oneHotVector)

        logging.debug(X_entity[0])
        logging.debug(X_left[0])
        logging.debug(X_right[0])

        # TODO: dynamically select maxlen
        X_entity = pad_sequences(X_entity, maxlen=entity_length)
        X_left = pad_sequences(X_left, maxlen=left_length, truncating='pre')
        X_right = pad_sequences(X_right, maxlen=right_length, truncating='post')
        y = np.asarray(y)

        return [X_left, X_entity, X_right], y

    @staticmethod
    def oneHotEncode(idxLabels, labelDim):
        return MyCorpus.oneHotEncode(idxLabels, labelDim)

    @staticmethod
    def oneHotDecode(y):
        return MyCorpus.oneHotDecode(y)

    @staticmethod
    def topK(y, topK=2):
        return MyCorpus.topK(y, topK)

    @staticmethod
    def threshold(y, threshold=0.5):
        return MyCorpus.threshold(y, threshold)

    @staticmethod
    def hybrid(y, threshold=0.5):
        return MyCorpus.hybrid(y, threshold)