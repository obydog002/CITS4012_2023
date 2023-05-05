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
            tf_idf = {}
            # initialise counter for the doc
            counter = Counter(sentence)
            # calculate total number of words in the doc
            total_num_words = len(sentence)
            # get each unique word in the doc
            for term in np.unique(sentence):
                # calculate Term Frequency 
                tf = counter[term]/total_num_words
                # calculate Document Frequency
                df = doc_freq[term]
                # calculate Inverse Document Frequency
                idf = math.log(N/(df+1))+1
                # calculate TF-IDF
                tf_idf[term] = tf*idf
            # then create list of tf-idf values
            sent_tf_idf = []
            for word in sentence:
                sent_tf_idf.append(tf_idf[word])
            doc_tf_idf.append(sent_tf_idf)
        return doc_tf_idf




    def ner_tag(doc):
        pass

    def word_match(doc):
        pass

    def lemmatize(word): # probably required for word_match()
        pass
