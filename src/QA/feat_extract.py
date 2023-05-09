# used for PoS tagging
import nltk
from nltk.tag import pos_tag_sents
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

# used to index the pos tags in FeatExt
index_of_pos = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5, 'JJ': 6, 'JJR': 7,
'JJS': 8, 'LS': 9, 'MD': 10, 'NN': 11, 'NNS': 12, 'NNP': 13, 'NNPS': 14,
'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18, 'RB': 19, 'RBR': 20, 'RBS': 21,
'RP': 22, 'SYM': 23, 'TO': 24, 'UH': 25, 'VB': 26, 'VBD': 27, 'VBG': 28, 
'VBN': 29, 'VBP': 30, 'VBZ': 31, 'WDT': 32, 'WP': 33, 'WP$': 34, 'WRB': 35, 
'.': 36, ',': 37, ':': 38}

# used for TF-IDF
import numpy as np
from collections import Counter
import math

# used for NER
import spacy
import en_core_web_sm
ner_model = en_core_web_sm.load()
import re

class FeatExt:

    def get_pos_tags(doc):
        list_of_pos = pos_tag_sents(doc)
        list_of_pos_indices = []
        for sentence in list_of_pos:
            sent_indices = []
            for tup in sentence:
                # give any unusual punctuation or symbols the unused index 39
                if tup[1] not in index_of_pos:
                    sent_indices.append(39)
                else:
                    sent_indices.append(index_of_pos[tup[1]])
            list_of_pos_indices.append(sent_indices)
        return list_of_pos_indices

    def question_pos(sentence):
        list_of_pos = pos_tag(sentence)
        sent_indices = []
        for tup in list_of_pos:
            # give any unusual punctuation or symbols the unused index 39
            if tup[1] not in index_of_pos:
                sent_indices.append(39)
            else:
                sent_indices.append(index_of_pos[tup[1]])
        return sent_indices


    def doc_freq_finder(doc):
        doc_freq = {}
        for sentence in doc:
          # get each unique word in the doc
          # and count the number of occurrences in the document
          for term in np.unique(sentence):
              try:
                  doc_freq[term] +=1
              except:
                  doc_freq[term] =1
        return doc_freq
    
    def tf_idf_fun(doc):
        doc_freq = FeatExt.doc_freq_finder(doc)
        # total number of sentences
        N = len(doc)
        doc_tf_idf = []
        for sentence in doc:
            # create a dictionary of tf-idf values
            tf_idf = {}
            counter = Counter(sentence)
            total_num_words = len(sentence)
            for term in np.unique(sentence):
                tf = counter[term]/total_num_words
                df = doc_freq[term]
                idf = math.log(N/(df+1))+1
                tf_idf[term] = tf*idf
            # then create list of tf-idf values using the dictionary
            sent_tf_idf = []
            for word in sentence:
                sent_tf_idf.append(tf_idf[word])
            doc_tf_idf.append(sent_tf_idf)
        return doc_tf_idf

    def ner_tag(sentence):
        # this MUST be run with non-tokenised sentences
        # we apply the same spacing regex as in the data_prep to try to ensure the
        # output will be the same length as the tokenised sentence
        spaced_sentence = re.sub('(?<=[^ ])(?=[.,!?()\'\"\-:;])|(?<=[.,!?()\'\"\-:;])(?=[^ ])|\s{2,}', r' ', sentence)
        ner_output = ner_model(spaced_sentence)
        # we run as for loop just in case ner model has split the sentence
        # into multiple sentences
        ner_iobs = []
        ner_types = []
        for sent in ner_output.sents:
            for word in sent:
                ner_iobs.append(word.ent_iob)
                ner_types.append(word.ent_type)
                # these are both numeric indices, rather than labels
                # so no further processing needs to be done
        return ner_iobs, ner_types

    def ner_tag_doc(document):
        # this MUST be run with non-tokenised sentences
        ner_iobs = []
        ner_types = []
        for sentence in document:
            ner_output = FeatExt.ner_tag(sentence[0])
            ner_iobs.append(ner_output[0])
            ner_types.append(ner_output[1])
        return ner_iobs, ner_types

    def word_match(doc):
        pass

    def lemmatize(word): # probably required for word_match()
        pass
