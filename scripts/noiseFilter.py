import pickle
import logging
from collections import defaultdict

# :: Logging level ::
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

pklPath = 'onto_pkl/'

def filtering():
    rawPrediction = pickle.load(open(pklPath + 'raw.pkl', 'rb'))

    predictionList = [
        pickle.load(open(pklPath + 'LSTM.pkl', 'rb')),
        pickle.load(open(pklPath + 'LSTMSingle.pkl', 'rb')),
        pickle.load(open(pklPath + 'HNM.pkl', 'rb')),
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

def generateNewTrain():
    import json
    with open('filteredTrainLabel.pkl', 'rb') as pklFile:
        filteredPrediction = pickle.load(pklFile)
    with open(pklPath + 'ontoLabel2idx.pkl', 'rb') as pklFile:
        label2idx = pickle.load(pklFile)
    idx2label = {v:k for k,v in label2idx.items()}
    with open('../data/OntoNotes/train.json', 'r') as inputFile:
        with open('../data/OntoNotes/train_filtered.json', 'w') as outputFile:
            sentIdx = 0
            for line in inputFile:
                jsonObj = json.loads(line)
                for mention in jsonObj['mentions']:
                    mention['labels'] = list(map(idx2label.get, filteredPrediction[sentIdx]))
                    sentIdx += 1
                outputFile.write(json.dumps(jsonObj) + '\n')

if __name__ == '__main__':
    # filtering()
    generateNewTrain()
