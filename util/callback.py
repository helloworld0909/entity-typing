import os
import logging
import keras
import numpy as np
import time
from util.corpus import MyCorpus
from eval.evaluation import evaluate


class MetricHistory(keras.callbacks.Callback):

    def __init__(self, X_test, y_test):
        super(MetricHistory, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.saveDir = time.strftime("%m_%d_%H_%M", time.localtime()) + '/'
        if not os.path.exists(self.saveDir):
            os.mkdir(self.saveDir)
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        y_prob = self.model.predict(self.X_test)
        np.save(self.saveDir + 'epoch{:0>2}.npy'.format(epoch), y_prob)

        y_predict = MyCorpus.hybrid(y_prob, threshold=0.5)

        predictions = MyCorpus.oneHotDecode(y_predict)
        ground_truth = MyCorpus.oneHotDecode(self.y_test)
        metrics = evaluate(predictions, ground_truth)
        print('acc, ma_f1, mi_f1: {}'.format(metrics[0], metrics[3], metrics[6]))
        with open(self.saveDir + 'metric.txt', 'a') as metricFile:
            metricFile.write('\t'.join(map(str, metrics)) + '\n')
        self.history.append(metrics)