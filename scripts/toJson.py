import logging
import pickle
import json
import toJsonUtil
from collections import defaultdict

# :: Logging level ::
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

filepath = '../data/baike/'

entity2types = toJsonUtil.getEntityTypes(filepath + 'zh_entity_types.txt')

with open(filepath + 'baike_desc.txt', 'r', encoding='utf-8') as inputFile:

    outputFile = open(filepath + 'data.json', 'w', encoding='utf-8')

    sentID = 0
    entity2idx = {}

    for line in inputFile:
        entity, _, summary = line.strip('n').split('\t')
        if entity in entity2types:
            summary = toJsonUtil.preprocessZH(summary)
            sents = toJsonUtil.segmentSent(summary)
            sentID = 0
            for sent in sents:
                if entity in sent:
                    jsonObj = {'senid': sentID}
                    sentID += 1

                    if entity not in entity2idx:
                        entity2idx[entity] = len(entity2idx)
                    jsonObj['entityid'] = entity2idx[entity]

                    labels = entity2types[entity]

                    mention = toJsonUtil.parseEntity(entity)
                    seg, startEndIndices = toJsonUtil.segmentWithMention(sent, mention)
                    jsonObj['tokens'] = seg
                    jsonObj['mentions'] = []
                    for start, end in startEndIndices:
                        jsonObj['mentions'].append({'start': start, 'end': end, 'labels': labels})
                    jsonStr = json.dumps(jsonObj, ensure_ascii=False)
                    outputFile.write(jsonStr + '\n')

                    logging.debug(jsonStr)

    outputFile.close()


