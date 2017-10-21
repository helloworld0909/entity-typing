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
