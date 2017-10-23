import os
import sys
import logging
import pickle
from keras.models import load_model

from eval.evaluation import evaluate
from models.lstm import lstm, lstmSingle
from models.hnm import hnm
from util.corpus import MyCorpus
from util.callback import MetricHistory

# :: Logging level ::
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

dataSetName = 'BBN'
trainFilePath = 'data/{}/train.json'.format(dataSetName)
testFilePath = 'data/{}/test.json'.format(dataSetName)

corpus = MyCorpus(filePathList=[trainFilePath, testFilePath])

def trainLSTM():
    global trainFilePath, testFilePath, corpus

    X_train, y_train = corpus.loadFile(filePath=trainFilePath)
    X_test, y_test = corpus.loadFile(filePath=testFilePath)
    model = lstm(corpus)
    metricHistory = MetricHistory(X_test, y_test)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, shuffle=True, callbacks=[metricHistory])

    y_prob = model.predict(X_train)
    y_predict = MyCorpus.hybrid(y_prob, threshold=0.5)

    predictions = MyCorpus.oneHotDecode(y_predict)
    with open('LSTM.pkl', 'wb') as outputFile:
        pickle.dump(predictions, outputFile, pickle.HIGHEST_PROTOCOL)

def trainLSTMSingle():
    global trainFilePath, testFilePath, corpus

    X_train, y_train = corpus.loadFileSingleSent(filePath=trainFilePath)
    X_test, y_test = corpus.loadFileSingleSent(filePath=testFilePath)
    model = lstmSingle(corpus)
    metricHistory = MetricHistory(X_test, y_test)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, shuffle=True, callbacks=[metricHistory])
    y_prob = model.predict(X_train)
    y_predict = MyCorpus.hybrid(y_prob, threshold=0.5)

    predictions = MyCorpus.oneHotDecode(y_predict)
    with open('LSTMSingle.pkl', 'wb') as outputFile:
        pickle.dump(predictions, outputFile, pickle.HIGHEST_PROTOCOL)

def trainHNM():
    global trainFilePath, testFilePath, corpus

    X_train, y_train = corpus.loadFile(filePath=trainFilePath)
    X_test, y_test = corpus.loadFile(filePath=testFilePath)
    model = hnm(corpus)
    metricHistory = MetricHistory(X_test, y_test)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, shuffle=True, callbacks=[metricHistory])

    y_prob = model.predict(X_train)
    y_predict = MyCorpus.hybrid(y_prob, threshold=0.5)

    predictions = MyCorpus.oneHotDecode(y_predict)
    with open('HNM.pkl', 'wb') as outputFile:
        pickle.dump(predictions, outputFile, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    trainLSTM()
    trainLSTMSingle()
    trainHNM()