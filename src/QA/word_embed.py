import gensim.downloader as api
glo100model = api.load("glove-wiki-gigaword-100")

# w2v300model = api.load("word2vec-google-news-300")
# this may be too large for our purposes, but is only english w2v model on gensim

class WordEmbed:

    def glove_embed_sent(sentence):
        embeds = []
        for word_tup in sentence:
            embeds.append(glo100model[word_tup[0].lower()])
        return embeds
    
    # currently tokenize_question_and_doc is returning a list of lists (one for each sentence)
    def glove_embed(doc):
        doc_embeds = []
        for sentence in doc:
            doc_embeds.append(glove_embed_sent(sentence))
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

    def pos_tag(doc):
        pass

    def tf_idf(doc):
        pass

    def ner_tag(doc):
        pass

    def word_match(doc):
        pass

    def lemmatize(word): # probably required for word_match()
        pass
