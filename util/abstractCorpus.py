from collections import defaultdict


class AbstractCorpus(object):

    def __init__(self):
        self.token2idx = {'PADDING': 0, 'UNKNOWN': 1}
        self.label2idx = {}
        self.feature2idx = defaultdict(lambda : {'PADDING': 0})
        self.wordEmbedding = []
        self.vocabSize = 0
        self.labelDim = 0


    def initMappings(self, filePathList):
        """
        Initialize self.token2idx, self.label2idx
        """
        pass


    def initWordEmbedding(self, filePath, dim):
        pass


    def initMaxTokenLength(self):
        pass

    def initMaxSentenceLength(self, filePathList):
        pass

    def loadFile(self, filePath):
        pass


