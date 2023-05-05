import numpy as np
from word_embed import WordEmbed
from feat_extract import FeatExt

# concatenate the embed and feature tags togethe
# could add future functionality to choose to exclude certain features for ablation
def embedder(document):
    doc_embeds = WordEmbed.glove_embed(document)
    pos_indices = FeatExt.get_pos_tags(document)
    tf_idf_nums = FeatExt.tf_idf_fun(document)
    embedded_document = []
    num_sents = len(document)
    for i in range(num_sents):
        num_words = len(document[i])
        embedded_sentence = []
        for j in range(num_words):
            embedded_word = np.concatenate((doc_embeds[i][j], np.array([pos_indices[i][j],tf_idf_nums[i][j]])))
            embedded_sentence.append(embedded_word)
        embedded_document.append(embedded_sentence)
    return embedded_document
