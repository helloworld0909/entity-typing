import logging
from keras.models import load_model

from eval.evaluation import evaluate
from models.lstm import lstm
from util.corpus import MyCorpus

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

model = lstm(corpus)
model.fit(X_train, y_train, epochs=5, batch_size=32, shuffle=True)
model.save('model.h5')
# model = load_model('model.h5')
y_predict = model.predict(X_test)

predictions = corpus.oneHotDecode(y_predict)
ground_truth = corpus.oneHotDecode(y_test)
accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate(predictions, ground_truth)
print('accuracy:', accuracy)
print('macro_precision, macro_recall, macro_f1:', macro_precision, macro_recall, macro_f1)
print('micro_precision, micro_recall, micro_f1:', micro_precision, micro_recall, micro_f1)
