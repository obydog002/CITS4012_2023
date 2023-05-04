import gensim.downloader as api
glo100model = api.load("glove-wiki-gigaword-100")

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

# w2v300model = api.load("word2vec-google-news-300")
# this may be too large for our purposes, but is only english w2v model on gensim

class WordEmbed:

    def glove_embed_sent(sentence):
        embeds = []
        for word_tup in sentence:
            embeds.append(glo100model[word_tup[0].lower()])
        return embeds
    
    def glove_embed(doc):
        doc_embeds = []
        for sentence in doc:
            doc_embeds.append(WordEmbed.glove_embed_sent(sentence))
        return doc_embeds

    # def w2v_embed_sent(sentence):
    #     embeds = []
    #     for word_tup in sentence:
    #         embeds.append(w2v300model[word_tup[0].lower()])
    #     return embeds
    
    # def w2v_embed(doc):
    #     doc_embeds = []
    #     for sentence in doc:
    #         doc_embeds.append(w2v_embed_sent(sentence))
    #     return doc_embeds

class FeatExt:

    def untuple(doc):
        # not needed anymore since I've changed data prep
        untuped_doc = []
        for sentence in doc:
            untuped_sent = []
            for tup in sentence:
                untuped_sent.append(tup[0])
            untuped_doc.append(untuped_sent)
        return untuped_doc
    
    def get_pos_tags(doc):
        list_of_pos = pos_tag_sents(doc)
        list_of_pos_tags = []
        for sentence in list_of_pos:
            sent_pos = []
            for tup in sentence:
                if tup[1] not in index_of_pos:
                    sent_pos.append(39)
                else:
                    sent_pos.append(index_of_pos[tup[1]])
            list_of_pos_tags.append(sent_pos)
        return list_of_pos_tags

    def question_pos(sentence):
        list_of_pos = pos_tag(sentence)

    def tf_idf(doc):
        pass

    def ner_tag(doc):
        pass

    def word_match(doc):
        pass

    def lemmatize(word): # probably required for word_match()
        pass
