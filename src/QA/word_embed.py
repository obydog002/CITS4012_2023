import numpy as np
import gensim.downloader as api
from nltk.translate.ribes_score import sentence_ribes
glo100model = api.load("glove-wiki-gigaword-100")

# w2v300model = api.load("word2vec-google-news-300")
# this may be too large for our purposes, but is only english w2v model on gensim

class WordEmbed:

    def glove_embed_sent(sentence):
        embeds = []
        for word in sentence:
            lword = word.lower()
            if lword in glo100model:
                embeds.append(glo100model[lword])
            else:
                embeds.append(np.zeros(100))
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
    