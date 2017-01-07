from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from PyDictionary import PyDictionary
from nltk.corpus import nps_chat as nchat
import nltk
from Utils import getNextTag

def OpinionQuestion(text, posTag):
    keywords = {'subiecti': [], 'criterii': []}

    for index in xrange(len(posTag)):
        if posTag[index][1] == 'WP' and posTag[index][0].lower() == 'what':
            # What is your opinion about X?
            nextTag, index2 = getNextTag(posTag, index, ['PRP$'])
            if nextTag is not None and nextTag[0].lower() == 'your':
                nextTag, index2 = getNextTag(posTag, index2, ['NN'])
                if nextTag is not None and nextTag[0].lower() == 'opinion':
                    nextTag, index2 = getNextTag(posTag, index2, ['NN','PRP','PRP$'])
                    if nextTag is not None:
                        keywords['criterii'].append(nextTag[0])
                        return True, keywords
            #what do you think about X?
            nextTag, index2 = getNextTag(posTag, index, ['PRP'])
            if nextTag is not None and nextTag[0].lower() == 'you':
                nextTag, index2 = getNextTag(posTag, index2, ['VB'])
                if nextTag is not None and nextTag[0].lower() == 'think':
                    nextTag, index2 = getNextTag(posTag, index2, ['NN','PRP','PRP$'])
                    if nextTag is not None:
                        keywords['criterii'].append(nextTag[0])
                        return True, keywords

        if posTag[index][1] == 'WRB' and posTag[index][0].lower() in ['when','where']:
            # When do you think school will begin?
            nextTag, index2 = getNextTag(posTag, index, ['PRP'])
            if nextTag is not None and nextTag[0].lower() == 'you':
                nextTag, index2 = getNextTag(posTag, index2, ['VB'])
                if nextTag is not None and nextTag[0].lower() == 'think':
                    nextTag, index2 = getNextTag(posTag, index2, ['NN','PRP', 'PRP$'])
                    if nextTag is not None:
                        nextTag2, index2 = getNextTag(posTag, index2, ['VB'])
                        if nextTag2 is not None:
                            keywords['subiecti'].append(nextTag[0])
                            keywords['criterii'].append(nextTag2[0])
                            return True, keywords

    return False, []
