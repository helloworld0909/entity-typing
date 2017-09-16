import logging
import math
from collections import defaultdict
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from util import preprocess
from util.abstractCorpus import AbstractCorpus


class CorpusEN(AbstractCorpus):

    def __init__(self):
        super(CorpusEN, self).__init__()

        self.char2idx = preprocess.getChar2idx()
        self.maxTokenLength = 0
        self.maxSentenceLength = 0
        self.tokenIdx2charVector = []
        self.charEmbedding = []

    def initTokenIdx2charVector(self, maxTokenLength):
        tokenIdx2charVector = []
        for token, idx in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
            if idx != 0:
                charVector = list(map(lambda c: self.char2idx.get(c, 1), token))  # 1 for UNKNOWN char
            else:
                charVector = [0]  # PADDING
            tokenIdx2charVector.append(charVector)

        self.tokenIdx2charVector = np.asarray(pad_sequences(tokenIdx2charVector, maxlen=maxTokenLength))
        logging.debug('tokenIdx2charVector[2]: ' + str(self.tokenIdx2charVector[2]))

    def initCharEmbedding(self, charEmbeddingDim):
        for _ in self.char2idx:
            limit = math.sqrt(3.0 / charEmbeddingDim)
            vector = np.random.uniform(-limit, limit, charEmbeddingDim)
            self.charEmbedding.append(vector)
        logging.info('charEmbedding: ' + str(charEmbeddingDim))


    def initWordEmbedding(self, filePath, dim=100):
        """
        The tokens in the word embedding matrix are uncased 
        """
        word2vector = preprocess.loadWordEmbedding(filePath, dim=dim)
        for token, idx in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
            if idx >= 2:
                token = token.lower()
            vector = word2vector.get(token, np.random.uniform(-0.25, 0.25, dim))
            self.wordEmbedding.append(vector)
        self.wordEmbedding = np.asarray(self.wordEmbedding)
        logging.debug('wordEmbedding[2]: ' + str(self.wordEmbedding[2]))
        logging.debug('wordEmbedding.shape: ' + str(self.wordEmbedding.shape))

    def loadFile(self, filePath):
        raise NotImplementedError('You must implement loadFile()')