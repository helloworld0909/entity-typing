import pickle
import logging
from collections import defaultdict

# :: Logging level ::
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

rawPrediction = pickle.load(open('rawBBN.pkl', 'rb'))

predictionList = [
    pickle.load(open('LSTM.pkl', 'rb')),
    pickle.load(open('LSTMSingle.pkl', 'rb')),
    pickle.load(open('HNM.pkl', 'rb')),
]

ignoreList = [{}] * len(predictionList)

for idx, prediction in enumerate(predictionList):
    for sentIdx, labelSet in prediction.items():
        rawSet = rawPrediction[sentIdx]
        ignoreSet = rawSet - labelSet
        ignoreList[idx][sentIdx] = ignoreSet

ignoreFusion = {}
for sentIdx in range(len(ignoreList[0])):
    fusionSet = ignoreList[0][sentIdx]
    for idx in range(1, len(ignoreList)):
        fusionSet = fusionSet & ignoreList[idx][sentIdx]
    ignoreFusion[sentIdx] = fusionSet

assert len(ignoreFusion) == len(rawPrediction)

filteredPrediction = {}
filteredCount = 0
for sentIdx in rawPrediction.keys():
    filteredPrediction[sentIdx] = rawPrediction[sentIdx] - ignoreFusion[sentIdx]
    if ignoreFusion[sentIdx]:
        filteredCount += 1
print(filteredCount)

with open('filteredTrainLabel.pkl', 'wb') as pklFile:
    pickle.dump(filteredPrediction, pklFile, pickle.HIGHEST_PROTOCOL)
