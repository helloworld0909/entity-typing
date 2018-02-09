import os
import sys
import logging
import pickle
from keras.models import load_model

from eval.evaluation import evaluate
from models.lstm import lstm, lstmSingle
from models.hnm import hnm, hnm_origin
from util.corpus import MyCorpus
from util.callback import MetricHistory, MetricHistorySoftmax

# :: Logging level ::
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

dataSetName = 'baike'
trainFilePath = 'data/{}/train.json'.format(dataSetName)
testFilePath = 'data/{}/test.json'.format(dataSetName)

corpus = MyCorpus(filePathList=[trainFilePath, testFilePath])

def getRawLabels():
    global trainFilePath, testFilePath, corpus
    _, y_train = corpus.loadFile(filePath=trainFilePath)
    predictions = MyCorpus.oneHotDecode(y_train)
    with open('raw.pkl', 'wb') as outputFile:
        pickle.dump(predictions, outputFile, pickle.HIGHEST_PROTOCOL)
    with open(dataSetName + 'Label2idx.pkl', 'wb') as outputFile:
        pickle.dump(corpus.label2idx, outputFile, pickle.HIGHEST_PROTOCOL)

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
    with open(metricHistory.saveDir + 'LSTM.pkl', 'wb') as outputFile:
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
    with open(metricHistory.saveDir + 'LSTMSingle.pkl', 'wb') as outputFile:
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
    with open(metricHistory.saveDir + 'HNM.pkl', 'wb') as outputFile:
        pickle.dump(predictions, outputFile, pickle.HIGHEST_PROTOCOL)

def trainHNMOrigin():
    global trainFilePath, testFilePath, corpus

    X_train, y_train = corpus.loadFile(filePath=trainFilePath)
    X_test, y_test = corpus.loadFile(filePath=testFilePath)
    model = hnm_origin(corpus)
    metricHistory = MetricHistorySoftmax(X_test, y_test)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1, shuffle=True, callbacks=[metricHistory])

if __name__ == '__main__':
    getRawLabels()
    # trainLSTM()
    # trainLSTMSingle()
    # trainHNM()