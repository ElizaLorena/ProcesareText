from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from PyDictionary import PyDictionary
from nltk.corpus import nps_chat as nchat
import nltk

NOUN_TYPES = ['NN', 'NNS', 'NNP', 'PRP']


def DifferenceBetween(text, posTag):

    cc_found = contains_cc(text)
    nouns_found = count_nouns(text)

    if not cc_found:
        return False, {}

    if nouns_found < 2:
        return False, {}

    sentence = feature_select(text)
    result = classifier.classify(sentence)

    keywords = {'subiecti': [], 'criterii': ['difference']}

    if result == 'matched':
        for word, t in reversed(posTag):
            if t in NOUN_TYPES and word not in keywords_1:
                keywords['subiecti'].append(word)

        return True, keywords
    else:
        return False, []


def create_classifier():

    sample_matched = [
        "What's the difference between an ocean and a sea?",
        "What is the difference between weather and climate?",
        "What would you say about the difference between atoms and elements?",
        "How would you compare lacrosse to football?",
        "How would you compare NetSuite vs Intacct?",
        "How many km are between Iasi and Hawaii?",
        "Tell me the difference between Obama and Trump",
        "What is the difference between me and you?",
        "Elaborate on why Trump is better than Obama"
    ]

    sample_unmatched = nchat.xml_posts()[:800]

    features_matched = []
    features_unmatched = []

    for text in sample_matched:
        features = (feature_select(text), 'matched')
        features_matched.append(features)

    for item in sample_unmatched:
        text = item.text
        features = (feature_select(text), 'unmatched')
        features_unmatched.append(features)

    train_features = features_matched + features_unmatched
    return nltk.NaiveBayesClassifier.train(train_features)


def feature_select(text):

    features = {}

    for word, t in pos_tag(nltk.word_tokenize(text)):
        features['contains(%s)' % word.lower()] = True
        if word in keywords_1:
            features['contains(keywords_1)'] = True
        if word in keywords_2:
            features['contains(keywords_2)'] = True

    features['noun_count'] = count_nouns(text)
    return features


def count_nouns(text):

    posTag = pos_tag(word_tokenize(text))
    nouns_found = 0

    for tag in posTag:
        word, t = tag
        if t in NOUN_TYPES and word not in keywords_1:
            nouns_found += 1

    return nouns_found


def contains_cc(text):

    posTag = pos_tag(word_tokenize(text))

    for tag in posTag:
        word, t = tag
        if word in keywords_2 and t == 'CC':
            return True
        if word == 'vs' and t in ['NN', 'JJ']:
            return True
        if word == 'to' and t == 'TO':
            return True
        if word == 'than' and t == 'IN':
            return True

    return False


dictionary = PyDictionary()

keywords_1 = ['difference', 'compare', 'evaluate']
keywords_1 = [(dictionary.synonym(w) + [w]) for w in keywords_1]
keywords_1 = sum(keywords_1, [])  # flatten
keywords_2 = ['and', 'vs', 'to']

classifier = create_classifier()
