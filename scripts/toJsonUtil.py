import logging
import re
import random
import json
import collections
import jieba
from util.preprocess import selectPaddingLength


def parseDBPediaUrl(url):
    template = '<http://dbpedia.org/ontology(.*)>'
    groups = re.search(template, url)
    if groups:
        return groups.group(1).upper()
    else:
        logging.warning('DBPedia url error: {}'.format(url))
        return url

def parseEntity(e):
    match = re.match(r'^(.+)（.+）$', e)
    if match:
        mention = match.group(1)
        if mention:
            return mention
        else:
            return e
    else:
        return e

def getEntityTypes(filename):
    entity2types = collections.defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            line = line.strip('\n')
            entity, t = line.split('\t')
            entity2types[entity].append(t)
    return entity2types

def preprocessZH(sent):
    sent = sent.strip('\n')
    sent = re.sub(r'</?a>', '', sent)
    return sent

def segmentSent(paragraph):
    sents = paragraph.split('|||')
    res = []
    for sent in sents:
        res.extend(re.findall(r"[^；。]+[；。]", sent))
    return [sent for sent in res if sent]

def segmentWord(sent):
    return list(jieba.cut(sent))

def segmentWithMention(sent, mention):
    res = []
    startEndIndices = []
    parts = sent.split(mention)
    entitySeg = segmentWord(mention)
    for idx, part in enumerate(parts):
        seg = segmentWord(part)
        res.extend(seg)
        if idx != len(parts) - 1:
            start = len(seg)
            end = start + len(entitySeg)
            res.extend(entitySeg)
            startEndIndices.append((start, end))
    return res, startEndIndices

def trainTestSplit(filename):
    trainFile = open('train.json', 'w', encoding='utf-8')
    testFile = open('test.json', 'w', encoding='utf-8')

    lenDist = collections.defaultdict(int)
    entityDist = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            jsonObj = json.loads(line)
            lenDist[len(jsonObj['tokens'])] += 1
            entityDist[jsonObj['entityid']] += len(jsonObj['mentions'])
    print('padding: %s' %selectPaddingLength(lenDist, ratio=0.999))
    with open(filename, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            jsonObj = json.loads(line)
            freq = entityDist[jsonObj['entityid']]
            if freq < 3:
                trainFile.write(line)
            else:
                testFile.write(line)

def checkOverlap(trainFilename, testFilename):
    trainEntitySet = set()
    testEntitySet = set()
    with open(trainFilename, 'r', encoding='utf-8') as trainFile:
        for line in trainFile:
            jsonObj = json.loads(line)
            trainEntitySet.add(jsonObj['entityid'])
    with open(testFilename, 'r', encoding='utf-8') as testFile:
        for line in testFile:
            jsonObj = json.loads(line)
            testEntitySet.add(jsonObj['entityid'])
    if len(trainEntitySet & testEntitySet) == 0:
        print('Pass')
    else:
        print(trainEntitySet & testEntitySet)


if __name__ == '__main__':

    checkOverlap('E:/python_workspace/entity-typing/toJson/train.json', 'E:/python_workspace/entity-typing/toJson/test.json')

    exit(0)

    trainTestSplit('E:/python_workspace/entity-typing/data/baike/data.json')

    exit(0)

    sentence = "本文是扯淡ゝ旳青春所著首发于<a>吸血鬼骑士</a>贴吧；溯月流年（娜殇雪）转发于晋江文学城的已完结动漫同人小说。"
    sentence = preprocessZH(sentence)
    sentences = segmentSent(sentence)
    print(sentences)

    entity = '吸血鬼骑士'
    for s in sentences:
        if entity in s:
            print(segmentWithMention(s, parseEntity(entity)))
    print(parseEntity(entity))
    print(parseEntity('陈蓓（郑州西亚斯国际学院教师）'))

    exit(0)

    import json
    senidFreq = collections.defaultdict(int)
    for line in open('E:/python_workspace/entity-typing/data/baike/data.json', 'r', encoding='utf-8'):
        jsonObj = json.loads(line)
        senidFreq[jsonObj['senid']] += 1
    print(senidFreq)

    exit(0)

    fselectType = open('E:/python_workspace/entity-typing/data/baike/selected types.txt', 'r', encoding='utf-8')

    fp = open('E:/python_workspace/entity-typing/data/baike/zh_entity_types_common.txt', 'r', encoding='utf-8')
    fout = open('E:/python_workspace/entity-typing/data/baike/zh_entity_types.txt', 'w', encoding='utf-8')

    typeDict = {}
    for line in fselectType:
        t, _ = line.strip().split('\t')
        raw_t = '/' + re.search(r'/([^/]+)$', t).group(1)
        typeDict[raw_t] = t

    fselectType.close()
    print(typeDict)
    print(len(typeDict))

    for line in fp:
        e, url = line.strip().split('\t')
        raw_t = parseDBPediaUrl(url)
        if raw_t in typeDict:
            t = typeDict[raw_t]
            fout.write(e + '\t' + t + '\n')

    fp.close()
    fout.close()


