import os
import sys
import logging
from keras.models import load_model

from eval.evaluation import evaluate
from models.lstmCN import lstmCN
from util.corpus import MyCorpus, MyCorpusCN
from util.callback import MetricHistory

# :: Logging level ::
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

dataSetName = 'baike'
trainFilePath = 'data/{}/data.json'.format(dataSetName)
# testFilePath = 'data/{}/test.json'.format(dataSetName)

corpus = MyCorpusCN(filePathList=[trainFilePath])
X_train, y_train = corpus.loadFile(filePath=trainFilePath)
# X_test, y_test = corpus.loadFile(filePath=testFilePath)

model = lstmCN(corpus)
# metricHistory = MetricHistory(X_test, y_test)
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, shuffle=True)