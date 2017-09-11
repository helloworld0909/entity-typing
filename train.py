import os
import sys
import logging
from keras.models import load_model

from eval.evaluation import evaluate
from models.lstm import lstm
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
X_train, y_train = corpus.loadFile(filePath=trainFilePath)
X_test, y_test = corpus.loadFile(filePath=testFilePath)


modelName = 'bilstm-twoHidden2-dropout.h5'

if len(sys.argv) > 1 and sys.argv[1] == 'eval':
    model = load_model(modelName)
else:
    if os.path.exists(modelName):
        model = load_model(modelName)
    else:
        model = lstm(corpus)
    metricHistory = MetricHistory(X_test, y_test)
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, shuffle=True, callbacks=[metricHistory])
    model.save(modelName)
    logging.info(str(metricHistory.history))

y_prob = model.predict(X_test)

y_predict = corpus.topK(y_prob, topK=2)

predictions = corpus.oneHotDecode(y_predict)
ground_truth = corpus.oneHotDecode(y_test)
accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate(predictions, ground_truth)
print('---- Final result ----')
print('accuracy:', accuracy)
print('macro_precision, macro_recall, macro_f1:', macro_precision, macro_recall, macro_f1)
print('micro_precision, micro_recall, micro_f1:', micro_precision, micro_recall, micro_f1)
