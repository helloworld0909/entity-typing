import numpy as np
import json


def statTypes(dataSetName):
    filenameList = ['data/%s/train.json' %dataSetName, 'data/%s/test.json' %dataSetName]

    def foo(filename):
        labelSet = set()
        with open(filename, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                jsonObj = json.loads(line)
                mentions = jsonObj['mentions']
                for m in mentions:
                    for t in m['labels']:
                        labelSet.add(t)
        print(filename + '\t' + str(len(labelSet)))
        return labelSet

    trainLabelSet = foo(filenameList[0])
    testLabelSet = foo(filenameList[1])

    totalLabelSet = trainLabelSet | testLabelSet
    notFoundLabelSet = testLabelSet - trainLabelSet & testLabelSet
    print('Total: %s' %len(totalLabelSet))
    print(notFoundLabelSet)
    return

def statEntities(dataSetName):
    filenameList = ['data/%s/train.json' % dataSetName, 'data/%s/test.json' % dataSetName]

    def foo(filename):
        idSet = set()
        with open(filename, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                jsonObj = json.loads(line)
                entityid = jsonObj['entityid']
                idSet.add(entityid)
        print(filename + '\t' + str(len(idSet)))
        return idSet

    for filename in filenameList:
        foo(filename)

def stat1():
    from util.corpus import MyCorpus
    from eval.evaluation import evaluate

    dataSetName = 'BBN'
    trainFilePath = 'data/{}/train.json'.format(dataSetName)
    testFilePath = 'data/{}/test.json'.format(dataSetName)

    corpus = MyCorpus(filePathList=[trainFilePath, testFilePath])
    X_train, y_train = corpus.loadFile(filePath=trainFilePath)
    X_test, y_test = corpus.loadFile(filePath=testFilePath)

    y=np.load('10_24_17_32/epoch17.npy')
    ground = MyCorpus.oneHotDecode(y_test)

    scores = []
    outputFile = open('a.txt', 'w', encoding='utf-8')
    for threshold in np.arange(0.3, 0.6, 0.01):
        y1=MyCorpus.threshold(y, threshold)
        y1=MyCorpus.oneHotDecode(y1)
        score = evaluate(y1, ground)
        scores.append((threshold, score))
        outputFile.write('%.3f\t%.3f\t%.3f\t%.3f\n' %(threshold, score[0], score[3], score[6]))

    print(scores)
    print('Max: ' + str(max(scores, key=lambda kv: sum([kv[1][0], kv[1][3], kv[1][6]]))))
    outputFile.close()

if __name__ == '__main__':
    statEntities('baike')