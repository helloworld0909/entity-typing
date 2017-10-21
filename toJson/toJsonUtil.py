import logging
import re
import collections
import jieba


def parseDBPediaUrl(url):
    template = '<http://dbpedia.org/ontology(.*)>'
    groups = re.search(template, url)
    if groups:
        return groups.group(1).upper()
    else:
        logging.warning('DBPedia url error: {}'.format(url))
        return ''

def parseEntity(e):
    match = re.match(r'^(.*)（.*）$', e)
    if match:
        return match.group(1)
    else:
        return e

def getEntityTypes(filename):
    entity2types = collections.defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            line = line.strip('\n')
            entity, url = line.split('\t')
            parseType = parseDBPediaUrl(url)
            if parseType:
                entity2types[entity].append(parseType)
    return entity2types

def preprocessZH(sent):
    sent = sent.strip('\n')
    sent = re.sub(r'</?a>', '', sent)
    return sent

def segmentSent(paragraph):
    sents = paragraph.split('|||')
    res = []
    for sent in sents:
        res.extend(re.split(r"[；。]", sent))
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

if __name__ == '__main__':
    sentence = "本文是扯淡ゝ旳青春所著首发于<a>吸血鬼骑士</a>贴吧；溯月流年（娜殇雪）转发于晋江文学城的已完结动漫同人小说。"
    sentence = preprocessZH(sentence)
    sentences = segmentSent(sentence)

    entity = '吸血鬼骑士'
    for s in sentences:
        if entity in s:
            print(segmentWithMention(s, parseEntity(entity)))
    print(parseEntity(entity))
    print(parseEntity('陈蓓（郑州西亚斯国际学院教师）'))

    fp = open('E:/python_workspace/entity-typing/data/baike/zh_entity_types_common.txt', 'r', encoding='utf-8')
    fout = open('E:/python_workspace/entity-typing/data/baike/type_frequency.txt', 'w', encoding='utf-8')

    from collections import defaultdict
    typeFreq = defaultdict(int)
    for line in fp:
        url = line.strip().split('\t')[-1]
        typeFreq[parseDBPediaUrl(url)] += 1
    for t, freq in sorted(typeFreq.items(), key=lambda tf: tf[1], reverse=True):
        fout.write(t + '\t' + str(freq) + '\n')