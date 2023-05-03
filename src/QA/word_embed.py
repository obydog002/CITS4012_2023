import gensim.downloader as api
glo100model = api.load("glove-wiki-gigaword-100")

import nltk
from nltk.tag import pos_tag_sents
nltk.download('averaged_perceptron_tagger')

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
                sent_pos.append(tup[1])
                # need to add step to convert tag into index before appending
            list_of_pos_tags.append(sent_pos)
        return list_of_pos_tags

    def tf_idf(doc):
        pass

    def ner_tag(doc):
        pass

    def word_match(doc):
        pass

    def lemmatize(word): # probably required for word_match()
        pass
