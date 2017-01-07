from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from PyDictionary import PyDictionary
from nltk.corpus import nps_chat as nchat
import nltk
import random
import textList
from DifferenceBetween import DifferenceBetween
from YesOrNo import YesOrNo

class ProcesareText:
    def __init__(self):
        # searchType (categoria din care face parte propozitia)
        self.searchType = None
        # inputType (tipul propozitiei)
        self.inputType = None
        # keyWords de forma [[(cuvant cheie, sinonim),(cuvant cheie, sinonim)],[Subiect1, Subiect2, ...]]
        self.keyWords = list()
        # full text
        self.filters = ['YesOrNo', 'DifferenceBetween', 'PersonalQuestion', 'MathQuestion', 'ChooseBetween', 'InfoAbout']
        # anaphora
        self.Subjects = list()
        self.PRP = [('it','he','she', 'him', 'her'), ('we', 'they', 'us', 'them')]
        self.PRPP = [('his', 'her', 'its', 'hers'), ('our', 'their')]
        # train to classify Question
        self.filters = ['YesOrNo', 'DifferenceBetween', 'PersonalQuestion', 'MathQuestion', 'ChooseBetween', 'InfoAbout']
        self.dictionary = PyDictionary()

        #train to classify Question
        self._setQuestionWorld()

    def changeAnaphoraSubjects(self):
        self.Subjects = self.getAllTags('NNP')

    def setAnaphoraSubjects(self):
        if self.Subjects:
            for i in xrange(len(self.posTag)):
                if self.posTag[i][1] == 'PRP':
                    if (self.posTag[i][0].lower() in self.PRP[0]) and len(self.Subjects[0]) == 1:
                        self.text = self.text.replace(self.posTag[i][0], self.Subjects[0][0], 1)
                        self.posTag[i] = tuple([self.Subjects[0][0], self.Subjects[1]])
                    elif (self.posTag[i][0].lower() in self.PRP[1]) and len(self.Subjects[0]) > 1:
                        posTag = self.posTag[i][0]
                        self.posTag[i] = tuple([self.Subjects[0][0], self.Subjects[1]])
                        subjects = self.Subjects[0][0]
                        for subject in self.Subjects[0][1:]:
                            self.posTag.insert(i,tuple([subject,self.Subjects[1]]))
                            subjects += ', ' + subject
                        self.text = self.text.replace(posTag, subjects, 1)
                elif self.posTag[i][1] == 'PRP$':
                    if (self.posTag[i][0].lower() in self.PRPP[0]) and len(self.Subjects[0]) == 1:
                        subject = self.Subjects[0][0] + "'s"
                        self.text = self.text.replace(self.posTag[i][0], subject, 1)
                        self.posTag[i] = tuple([subject, self.Subjects[1]])
                    elif (self.posTag[i][0].lower() in self.PRPP[1]) and len(self.Subjects[0]) > 1:
                        posTag = self.posTag[i][0]
                        self.posTag[i] = tuple([self.Subjects[0][0], self.Subjects[1]])
                        subjects = self.Subjects[0][0] + "'s"
                        for subject in self.Subjects[0][1:]:
                            self.posTag.insert(i,tuple([subject,self.Subjects[1]]))
                            subjects += ", " + subject + "'s"
                        self.text = self.text.replace(posTag, subjects, 1)
        self.changeAnaphoraSubjects()

    def _clearParam(self):
        self.originaltext = None
        self.text = None
        self.searchType = None
        self.inputType = None
        self.keyWords = list()

    def setText(self, text):
        self._clearParam()
        self.originaltext = text
        self.text = self.errorSyntaxText(text)
        self.setTags()

    def _setQuestionWorld(self):
        self.dictionary = PyDictionary()
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
        listp = [',', '.', ';', '?', '!', '"', ":"]
        for p in listp:
            text = text.replace(p,' ')
        return text

    def setTags(self):
        # fiecare cuvant din propozitie cu partea s-a de vorbire (NN-subiect,...)
        self.posTag = pos_tag(word_tokenize(self.text))
        # self.poSentenceTag = pos_tag(sentence_tokenize(self.text))

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

    def getNextStructureTag(self, index, whatIs, direction=1):
        try:
            i = index
            while(i < len(self.posTag) and i > -1):
                i += direction
                tag = list(self.posTag[i])
                if self.posTag[i][1] in whatIs:
                    i += direction
                    while(i < len(self.posTag) and i > -1):
                        if self.posTag[i][1] not in whatIs:
                            return tuple(tag)
                        tag[0] += self.posTag[i][0]
                        i += direction
                    return tuple(tag)
            return None
        except Exception:
            return None

    def getAllTags(self, whatIs):
        allTags = list()
        for tag in self.posTag:
            if tag[1] == whatIs:
                allTags.append(tag[0])
        return [allTags, whatIs]

    def findStrings(self, stringsList):
        for i in xrange(len(self.posTag)):
            if self.posTag[i][0].lower() in stringsList:
                return i+1
        return 0

    def _setkeyWordsCriteriu(self, criteriu):
        try:
            if type(self.keyWords[0]) == list:
                pass
            elif type(self.keyWords[0] != list):
                self.keyWords[0] = list()
        except Exception:
            self.keyWords.append(list())
        self.keyWords[0].append((criteriu, str(self.getSynonym(criteriu))))

    def _setkeyWordsSubiecti(self, listaSubiecti):
        try:
            if type(self.keyWords[1]) == list:
                pass
            elif type(self.keyWords[1] != list):
                self.keyWords[1] = list()
        except Exception:
            while (len(self.keyWords) < 2):
                self.keyWords.append(list())
        for subiect in listaSubiecti:
            self.keyWords[1].append(subiect)

    #tag[0] - life
    #tag[1] - NN
    def PersonalQuestion(self):
        if self.findStrings(['your','about']):
            # Tell me your name.
            # What is your name?
            # Is Cata your name?
            # What about your name?
            # Tell me one of your favorite movie.
            index = self.findStrings(['your','about']) - 1
            nextTag = self.getNextStructureTag(index, ['NN','NNP'])
            if nextTag is not None:
                if self.findStrings(['tell', 'what', 'who']):
                    self._setkeyWordsCriteriu(nextTag[0])
                    if self.findStrings(['about']):
                        i = self.findStrings(['about'])-1
                        self._setkeyWordsSubiecti([self.getNextStructureTag(i,['NN','NNP'])])
                    return True
                elif self.findStrings(['where']):
                    self._setkeyWordsCriteriu('location')
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
                elif self.findStrings(['when']):
                    self._setkeyWordsCriteriu('time')
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
                elif self.findStrings(['why']):
                    self._setkeyWordsCriteriu('opinion')
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
                elif self.findStrings(['is']):
                    i = self.findStrings(['is']) - 1
                    if i > 0 and (self.posTag[i - 1][1] == 'NN' or self.posTag[i - 1][1] == 'NNP' or self.posTag[i - 1][1] == 'JJ'):
                        self._setkeyWordsCriteriu(nextTag[0])
                        self._setkeyWordsSubiecti([self.posTag[i-1][0],])
                        return True
                    elif i < len(self.posTag)-1 and (self.posTag[i + 1][1] == 'NN' or self.posTag[i + 1][1] == 'NNP' or self.posTag[i + 1][1] == 'JJ'):
                        self._setkeyWordsCriteriu(nextTag[0])
                        self._setkeyWordsSubiecti([self.posTag[i+1][0],])
                        return True
                elif self.findStrings(['do']):
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
        elif self.findStrings(['you']):
            index = self.findStrings(['you'])-1
            if self.findStrings(['what']):
                nexTag = self.getNextStructureTag(index, 'VB')
                if nexTag is not None:
                    self._setkeyWordsCriteriu(nexTag[0])
                    nextTag = self.getNextStructureTag(index, ['NN','NNS','NNP'])
                    if nextTag is not None:
                        self._setkeyWordsSubiecti([nextTag[0]])
                    else:
                        nextTag = self.getNextStructureTag(index, ['NN','NNS','NNP'], direction=-1)
                        if nextTag is not None:
                            self._setkeyWordsSubiecti([nextTag[0]])
                    return True
            elif self.findStrings(['who']):
                if self.findStrings(['are']):
                    self._setkeyWordsCriteriu('identity')
                    return True
            elif self.findStrings(['where']):
                nextTag = self.getNextStructureTag(index, ['VB','VBG','IN'])
                if nextTag[0] is not None:
                    self._setkeyWordsCriteriu('location')
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
                elif nextTag[1] == 'VB' or nextTag[1] == 'VBG':
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
            elif self.findStrings(['why']):
                self._setkeyWordsCriteriu('opinion')
                nexTag = self.getNextStructureTag(index, ['VB','IN'])
                if nexTag is not None:
                    self._setkeyWordsCriteriu(nexTag[0])
                    nextTag = self.getNextStructureTag(index, ['NN', 'NNS', 'NNP'])
                    if nextTag is not None:
                        self._setkeyWordsSubiecti([nextTag[0]])
                    else:
                        nextTag = self.getNextStructureTag(index, ['NN', 'NNS', 'NNP'], direction=-1)
                        if nextTag is not None:
                            self._setkeyWordsSubiecti([nextTag[0]])
                    return True
            elif self.findStrings(['how']):
                if self.findStrings(['old']):
                    self._setkeyWordsCriteriu('age')
                    return True
                nextTag = self.getNextStructureTag(index, ['NN','NNS', 'NNP'],direction=-1)
                if nextTag is not None:
                    self._setkeyWordsCriteriu(nextTag[0])
                    return True
            elif self.findStrings(['like','have','work']):
                nexTag = self.getNextStructureTag(index, 'VB')
                if nexTag is not None:
                    self._setkeyWordsCriteriu(nexTag[0])
                    nextTag = self.getNextStructureTag(index, ['NN', 'NNS', 'NNP'])
                    if nextTag is not None:
                        self._setkeyWordsSubiecti([nextTag[0]])
                    else:
                        nextTag = self.getNextStructureTag(index, ['NN', 'NNS', 'NNP'], direction=-1)
                        if nextTag is not None:
                            self._setkeyWordsSubiecti([nextTag[0]])
                    return True
            elif (index>0 and self.posTag[index-1][0]=='are') or (index<len(self.posTag)-1 and self.posTag[index+1][0]=='are'):
                nextTag = self.getNextStructureTag(index, 'JJ')
                if nextTag is not None:
                    self._setkeyWordsCriteriu('compliment')
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

    def ChooseBetween(self):
        for index in xrange(len(self.posTag)):
            # Who is older? Obama, Putin, Merkel or Churchill?
            # Who is older between Obama, Putin, Merkel and Churchill?
            # Which is solid between rock and vodka?
            if self.posTag[index][1] == 'VBZ' and self.posTag[index][0] == 'is' and self.posTag[index + 1][1] in ['JJR',
                                                                                                                  'JJ']:
                nextTag = self.getNextTag(index, ['JJR', 'JJ'])
                if nextTag is not None:
                    self._setkeyWordsCriteriu(nextTag[0])
                    i = index
                    leng = len(self.posTag)
                    while i < leng:
                        i += 1
                        nextTag = self.getNextTag(i, ['NNP', 'IN', 'CC', 'NN'])
                        if nextTag is not None:
                            if nextTag[1] not in ['NNP', 'NN']:
                                leng -= 1
                            else:
                                self._setkeyWordsSubiecti(nextTag[0])
                return True

            # Who has more/less
            elif self.posTag[index][1] == 'VBZ' and self.posTag[index][0].lower() == 'has' and self.posTag[index + 1][
                1] in ['JJR', 'JJ']:
                nextTag = self.getNextTag(index, ['JJR', 'JJ'])
                if nextTag is not None:
                    if nextTag[0] in ['more', 'less', 'fewer', 'greater']:
                        self._setkeyWordsCriteriu(nextTag[0])
                        noun = self.getNextTag(index + 1, ['NN', 'NNS', 'JJ'])
                        self._setkeyWordsCriteriu(noun[0])
                    i = index + 1
                    leng = len(self.posTag)
                    while i < leng:
                        i += 1
                        nextTag = self.getNextTag(i, ['NNP', 'IN', 'CC', 'NN'])
                        if nextTag is not None:
                            if nextTag[1] not in ['NNP', 'NN']:
                                leng -= 1
                            else:
                                self._setkeyWordsSubiecti(nextTag[0])
                return True
        return False

    def MathQuestion(self):
        for index in xrange(len(self.posTag)):
            # What is the root of the (equation)..? ex:2*x+4=0
            if self.posTag[index][1] == 'NN' and self.posTag[index][0].lower() == 'root':
                nextTag = self.getNextTag(index, ['NN'])
                if nextTag != None:
                    if (nextTag[0] == 'equation'):
                        self._setkeyWordsCriteriu('root of equation')
                        listaSubiecti = []
                        subiect = self.getNextTag(index, ['CD'])
                        subiect = subiect[0]
                        listaSubiecti.append(subiect)
                        self._setkeyWordsSubiecti(listaSubiecti)
                        subiect=self.getNextTag(index,['CD'])
                        subiect=subiect[0]
                        self._setkeyWordsSubiecti(subiect)
                        return True

            # pentru operatii matematice simple formate doar din 2 termeni (ex:What is the result of the 9+7?
            if self.posTag[index][1] == 'NN' and self.posTag[index][0].lower() == 'result':
                nextTag = self.getNextTag(index, ['IN'])
                if nextTag != None:
                    if (nextTag[0] == 'of'):
                        self._setkeyWordsCriteriu('result of arithmetic operation')
                        subiect = self.getNextTag(index, ['CD'])
                        subiect = subiect[0]
                        self._setkeyWordsSubiecti(subiect)
                        return True
            # What is the value of PI?
            if self.posTag[index][1] == 'IN' and self.posTag[index][0].lower() == 'of':
                nextTag = self.getNextTag(index, ['NNP'])
                if nextTag != None:
                    if (nextTag[0].lower() == 'pi'):
                        self._setkeyWordsCriteriu('value of pi')
                        listaSubiecti = []
                        subiect = self.getNextTag(index, ['NNP'])
                        subiect = subiect[0]
                        listaSubiecti.append(subiect)
                        self._setkeyWordsSubiecti(listaSubiecti)
                        subiect = self.getNextTag(index, ['NNP'])
                        subiect=subiect[0]
                        self._setkeyWordsSubiecti(subiect)
                        return True
            # What is the integral of ...?
            if self.posTag[index][1] == 'JJ' and self.posTag[index][0].lower() == 'integral':
                nextTag = self.getNextTag(index, ['CD'])
                if nextTag != None:
                    self._setkeyWordsCriteriu('integral of')
                    subiect = self.getNextTag(index, ['CD'])
                    subiect = subiect[0]
                    self._setkeyWordsSubiecti(subiect)
                    return True
            # What is half of x?
            if self.posTag[index][1] == 'NN' and self.posTag[index][0].lower() == 'half':
                nextTag = self.getNextTag(index, ['CD'])
                if nextTag != None:
                    self._setkeyWordsCriteriu('half of')
                    subiect = self.getNextTag(index, ['CD'])
                    subiect = subiect[0]
                    self._setkeyWordsSubiecti(subiect)
                    return True
            # What is sqrt/radical of x?
            if self.posTag[index][1] == 'NN' and self.posTag[index][0].lower() == 'sqrt' or \
                            self.posTag[index][0].lower() == 'radical':
                nextTag = self.getNextTag(index, ['CD'])
                if nextTag != None:
                    self._setkeyWordsCriteriu('sqrt of')
                    subiect = self.getNextTag(index, ['CD'])
                    subiect = subiect[0]
                    self._setkeyWordsSubiecti(subiect)
                    return True
            # What number comes after x?
            if self.posTag[index][1] == 'NN' and self.posTag[index][0].lower() == 'number':
                nextTag = self.getNextTag(index, ['IN'])
                if nextTag != None:
                    if (nextTag[0].lower() == 'after'):
                        self._setkeyWordsCriteriu('number after x')
                        listaSubiecti = []
                        subiect = self.getNextTag(index, ['CD'])
                        subiect = self.getNextTag(index, ['NN'])
                        subiect = subiect[0]
                        self._setkeyWordsSubiecti(subiect)
                        return True
            # What number comes before x?
            if self.posTag[index][1] == 'NN' and self.posTag[index][0].lower() == 'number':
                nextTag = self.getNextTag(index, ['IN'])
                if nextTag != None:
                    if (nextTag[0].lower() == 'before'):
                        self._setkeyWordsCriteriu('number before x')
                        listaSubiecti = []
                        subiect = self.getNextTag(index, ['CD'])
                        subiect = self.getNextTag(index, ['NN'])
                        subiect = subiect[0]
                        self._setkeyWordsSubiecti(subiect)
                        return True
        return False

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
        for filterName in self.filters:
            filterWasImported = filterName in globals()
            if filterWasImported:
                f = globals()[filterName]
                r, v = f(self.text, self.posTag)
                if r:
                    self.searchType = filterName
                    #self.searchType = self.filters.index(filterName)
                    for item in v['criterii']:
                        self._setkeyWordsCriteriu(item)
                    for item in v['subiecti']:
                        self._setkeyWordsSubiecti(item)
                    break
            else:
                r = getattr(self, filterName, lambda: False)()
                if r:
                    self.searchType = filterName
                    #self.searchType = self.filters.index(filterName)
                    break

    def setInputType(self):
        self.inputType = self.classify_sentence(self.text)

procesare = ProcesareText()

sample_texts = textList.differenceExamples

for text in sample_texts:
    procesare.setText(text)
    procesare.setAnaphoraSubjects()
    procesare.setFilter()
    procesare.setInputType()
    print  ('Original Text: %r') %procesare.originaltext
    print ('Text: %r') %procesare.text
    print ('Tags: %r') %procesare.posTag
    print ('(Sentence)Tags: %r') %procesare.posTag
    print ('Search Type: %r') %procesare.searchType
    print ('Key Words (criterii,subiecti): %r') %procesare.keyWords
    print ('Question: %r') %procesare.inputType
    print
