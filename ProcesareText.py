from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from PyDictionary import PyDictionary
from nltk.corpus import nps_chat as nchat
import nltk
import random
from DifferenceBetween import DifferenceBetween


class ProcesareText:
    def __init__(self):
        # searchType (categoria din care face parte propozitia)
        self.searchType = None
        # inputType (tipul propozitiei)
        self.inputType = None
        # keyWords de forma [[(cuvant cheie, sinonim),(cuvant cheie, sinonim)],[Subiect1, Subiect2, ...]]
        self.keyWords = []
        # full text
        self.text = self.errorSyntaxText("How many km are between Iasi and Suceava?")
        self.filters = ['YesOrNo', 'PersonalQuestion', 'ChooseBetween', 'DifferenceBetween', 'MathQuestion', 'InfoAbout']
        self.dictionary = PyDictionary()

        #train to classify Question
        self._setQuestionWorld()

    def _setQuestionWorld(self):
        posts = nchat.xml_posts()[:10000]
        featuresets = [(self.dialogue_act_features(post.text), post.get('class')) for post in posts]
        size = int(len(featuresets) * 0.1)
        train_set, test_set = featuresets[size:], featuresets[:size]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        self.classifier.labels()

    def dialogue_act_features(self, post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains(%s)' % word.lower()] = True
        return features

    def classify_sentence(self, sentence):
        final_test_sentence = self.dialogue_act_features(sentence)
        result = self.classifier.classify(final_test_sentence)
        if (result == "whQuestion") or (result == "ynQuestion"):
            return True
        else:
            return False

    # se scot semnele de punctuatie
    def errorSyntaxText(self, text):
        listp = [',', '.', ';', '?', '!', "'", '"', ":"]
        for p in listp:
            text = text.replace(p,' ')
        return text

    def getTags(self):
        # fiecare cuvant din propozitie cu partea s-a de vorbire (NN-subiect,...)
        self.posTag = pos_tag(word_tokenize(self.text))

    def getSynonym(self, word):
        listSynonyms = self.dictionary.synonym(self.errorSyntaxText(word))
        if listSynonyms:
            return random.choice(listSynonyms)

    # index reprezinta pozitia de unde incepe cautarea unui cuvant cu partea de vorbire ce se gaseste in whatIs
    # in caz de sunt mai multe una dupa alta se va returna ultimul cuvant
    # Functie creata pentru PesonalQuestion
    def getNextTag(self, index, whatIs):
        try:
            i = index
            while(i < len(self.posTag)):
                i += 1
                tag = self.posTag[i]
                if tag[1] in whatIs:
                    i += 1
                    while(i<len(self.posTag)):
                        if self.posTag[i] not in whatIs:
                            break
                        else:
                            tag = self.posTag[i]
                            i+=1
                    return tag
            return None
        except Exception:
            return None

    def _setkeyWordsCriteriu(self, criteriu):
        try:
            if type(self.keyWords[0]) == list:
                pass
        except Exception:
            self.keyWords.append(list())
        self.keyWords[0].append([criteriu, str(self.getSynonym(criteriu))])

    def _setkeyWordsSubiecti(self, listaSubiecti):
        try:
            if type(self.keyWords[1]) == list:
                pass
        except Exception:
            self.keyWords.append(list())
        for subiect in listaSubiecti:
            self.keyWords[1].append(subiect)

    #tag[0] - life
    #tag[1] - NN
    def PersonalQuestion(self):
        for index in xrange(len(self.posTag)):
            if self.posTag[index][1] == 'PRP$' and self.posTag[index][0].lower() == 'your':
                # Tell me your name
                nextTag = self.getNextTag(index, ['NN'])
                if nextTag is not None:
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
            elif self.posTag[index][1] == 'PRP' and self.posTag[index][0].lower() == 'you':
                # Where do you live? You are funny.
                nextTag = self.getNextTag(index, ['VBP','VBD','VB','JJ'])
                if nextTag != None:
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
            elif self.posTag[index][1] == 'WRB' and self.posTag[index][0].lower() == 'where'\
                    or self.posTag[index][0].lower() == 'how' or self.posTag[index][0].lower() == 'who':
                if self.text.find('you')!=-1 and self.text.find('are')!=-1:
                    self._setkeyWordsCriteriu('you')
                    return True
                # Where are you from?
                nextTag = self.getNextTag(index, 'IN')
                if nextTag != None and nextTag[0].lower() == 'from':
                    self._setkeyWordsCriteriu('location')
                    return True
                #How old are you?
                nextTag = self.getNextTag(index, 'JJ')
                if nextTag != None:
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
        return False

    ####### START INFO ABOUT #######
    def _getSubjectList(self, start, end):
        try:
            result = list()
            for tags in xrange(start, end):
                if self.posTag[tags][1] not in ['VBZ', 'VBD', 'VBN', 'VBP', 'IN', 'DT', 'WP', 'WDT', 'WRB', 'CC'] and self.posTag[tags][0].lower() != "which":
                    result.append(self.posTag[tags][0])
            return result
        except:
            return None

    def _getCriteria(self,start,end):
        try:
            result = ""
            for tags in xrange(start, end):
                if self.posTag[tags][1] in ['VBZ', 'VBD', 'VBN', 'VBP']:
                    result += self.posTag[tags][0]
                    result += " "
            return result
        except:
            return None

    def InfoAbout(self):
        if self.searchType is not None:
            return False

        for index in xrange(len(self.posTag)):
            if self.posTag[index][1] == 'WP' and self.posTag[index][0].lower() in ['what', 'who']:
                # What is onion?
                # Who won six consecutive Wimbledon singles titles in the 1980s?
                nextTag = self._getCriteria(0, len(self.posTag))
                if nextTag is not None or nextTag != "":
                    nextNextTag = self._getSubjectList(0, len(self.posTag))
                    if nextNextTag is not None:
                        self._setkeyWordsCriteriu(nextTag)
                        self._setkeyWordsSubiecti(nextNextTag)
                        return True

            elif self.posTag[index][1] == 'WRB' and self.posTag[index][0].lower() in ['where', 'how', 'why']:
                # Where is/was ... ?
                # How many arms/tentacles/limbs does a squid have?
                nextTag = self._getCriteria(0, len(self.posTag))
                if nextTag is not None or nextTag != "":
                    nextNextTag = self._getSubjectList(0, len(self.posTag))
                    if nextNextTag is not None:
                        if self.posTag[index][0].lower() == 'where':
                            self._setkeyWordsCriteriu('location')
                        if self.posTag[index][0].lower() == 'why':
                            self._setkeyWordsCriteriu('reason')
                        self._setkeyWordsCriteriu(nextTag)
                        self._setkeyWordsSubiecti(nextNextTag)
                        return True

            elif self.posTag[index][1] in ['WDT', 'JJ', 'NNP'] and self.posTag[index][0].lower() in ['which']:
                # Which hills divide England from Scotland?
                nextTag = self._getCriteria(0, len(self.posTag))
                if nextTag is not None or nextTag != "":
                    nextNextTag = self._getSubjectList(0, len(self.posTag))
                    if nextNextTag is not None:
                        self._setkeyWordsCriteriu(nextTag)
                        self._setkeyWordsSubiecti(nextNextTag)
                        return True
        return False
    ####### END INFO ABOUT #######

    def setFilter(self):

        self.getTags()

        for filterName in self.filters:

            filterWasImported = filterName in globals()

            if filterWasImported:
                f = globals()[filterName]
                r, v = f(self.text, self.posTag)

                if r:
                    self.searchType = filterName

                    for item in v['criterii']:
                        self._setkeyWordsCriteriu(item)

                    for item in v['subiecti']:
                        self._setkeyWordsSubiecti(item)

                    break
            else:
                r = getattr(self, filterName, lambda: False)()

                if r:
                    self.searchType = filterName
                    break

    def setInputType(self):
        self.inputType = self.classify_sentence(self.text)

procesare = ProcesareText()
procesare.setFilter()
procesare.setInputType()
print ('Text: %r') %procesare.text
print ('Tags: %r') %procesare.posTag
print ('Search Type: %r') %procesare.searchType
print ('Key Words (criterii,subiecti): %r') %procesare.keyWords
print ('Question: %r') %procesare.inputType
