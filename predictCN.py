import logging
import numpy as np
import pickle
from keras.models import load_model
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
trainFilePath = 'data/{}/train.json'.format(dataSetName)
filePath = 'data/{}/all_data.json'.format(dataSetName)

corpus = MyCorpusCN(filePathList=[trainFilePath])

with open('CNLabel2idx', 'wb') as pklFile:
    pickle.dump(corpus.label2idx, pklFile)

model = load_model('model.h5')

X, _ = corpus.loadFileAdditional(filePath)
y_predict = model.predict(X, verbose=1)
np.save('allCNProb.npy', y_predict)

