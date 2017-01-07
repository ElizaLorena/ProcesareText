from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from PyDictionary import PyDictionary
from nltk.corpus import nps_chat as nchat
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from Utils import getNextTag

lemmatizer = WordNetLemmatizer()


def YesOrNo(text, posTag):
    keywords = {'subiecti': [], 'criterii': []}

    for index in xrange(len(posTag)):
        tagPair = posTag[index]

        # 1.
        if(
            (
                tagPair[1]=="VBD" 
                or 
                tagPair[1]=="VBP" 
                or 
                tagPair[1]=="VBZ" 
            )
            # and 
            # lemmatizer.lemmatize(tagPair[0], 'v') == "be"
        ):
            # print("\n\nVERB FOUND! : " + tagPair[0] + "\n\n")
            
            
            subjectTagPair = getNextTag(posTag, index, ["NN","NNS","NNP","NNPS","PRP"])

            print(subjectTagPair)

            subject = subjectTagPair[0]
            subjectIndex = subjectTagPair[1]

            if subject is not None:
                # print("\n\nSUBJECT AFTER VERB FOUND! : " + subject[0] + "\n\n")

                # print(subjectIndex)
                criterion = getNextTag(posTag, subjectIndex, ["NN","NNS","NNP","NNPS","JJ","JJS","JJR"])[0]

                if criterion is not None:

                    # print("\n\nOBJECT AFTER SUBJECT AFTER VERB FOUND! : " + criterion[0] + "\n\n")

                    keywords['criterii'].append(criterion[0])
                    keywords['subiecti'].append(subject[0])

                    # print(keywords)

                    return True, keywords
        #END 1. 
        # 
        # 2.   
    return False, keywords
    
sample_questions = [
        #1. If the main verb of the sentence is "to be", simply invert the subject and the verb to be:
        "Are they American?", # to be - subject - object
        "Is New York nice?", # from "New York is nice"
        "Was he pepsi?",

        #2. If the sentence includes a main verb and another or other helping (auxiliary) verb(s), invert the subject and the (first) helping (auxiliary) verb.
        "Are they visiting Paris",
        "Has Nancy been working all night long?", #from "Nancy has been working all night long."

        #3. If the sentence includes a verb which is not the verb "to be" and doesn't include a helping (auxiliary) verb, the transformation is more complex.
        #3.a. If the verb is in the present tense, add either do or does and put the main verb in its base form: 

        #3.a.1. 'do' if the subject is the first person singular, second person singular, first person plural, second person plural and third person plural (I, you, we, they)
        "Do you like apples?", #from "You like apples."
        "Do they go to a high school?", #from "They go to a high school."

        #3.a.2. 'does' if the subject is the third person singular (he, she, it).
        "Does Nancy read a lot?",
        "Does he hate basketball?",

        #3.b. If the verb is in the past tense, add did and put the main verb in its base form:
        "Did John discover the truth?",
        "Did the chicken cross the road?"
    ]