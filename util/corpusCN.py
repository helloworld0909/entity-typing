import logging
import math
from collections import defaultdict
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from util import preprocess
from util.abstractCorpus import AbstractCorpus


class CorpusCN(AbstractCorpus):

    def __init__(self):
        super(CorpusCN, self).__init__()

    def initWordEmbedding(self, dim=100):
        """
        Randomly initialize word embedding
        """
        np.random.seed(1)
        self.wordEmbedding = np.empty(shape=(len(self.token2idx), dim))
        for idx in range(len(self.token2idx)):
            vector = np.random.uniform(-0.25, 0.25, dim)
            self.wordEmbedding[idx] = vector

        logging.debug('wordEmbedding[2]: ' + str(self.wordEmbedding[2]))
        logging.debug('wordEmbedding.shape: ' + str(self.wordEmbedding.shape))