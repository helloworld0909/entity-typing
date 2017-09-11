import logging
import keras
from util.corpus import MyCorpus
from eval.evaluation import evaluate


class MetricHistory(keras.callbacks.Callback):

    def __init__(self, X_test, y_test):
        super(MetricHistory, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        y_prob = self.model.predict(self.X_test)

        y_predict = MyCorpus.hybrid(y_prob, threshold=0.5)

        predictions = MyCorpus.oneHotDecode(y_predict)
        ground_truth = MyCorpus.oneHotDecode(self.y_test)
        metrics = evaluate(predictions, ground_truth)
        logging.info('acc: {}'.format(metrics[0]))
        logging.info('macro_precision, macro_recall, macro_f1: {}, {}, {}'.format(*metrics[1:4]))
        logging.info('micro_precision, micro_recall, micro_f1: {}, {}, {}'.format(*metrics[4:]))
        self.history.append(metrics)