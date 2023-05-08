import numpy as np
from word_embed import WordEmbed
from feat_extract import FeatExt

# concatenate the embed and feature tags together
# could add future functionality to choose to exclude certain features for ablation

class EmbedAndConcat:
    def doc_concat(document, with_ner = False, doc_ner_tags = None):
        doc_embeds = WordEmbed.glove_embed(document)
        pos_indices = FeatExt.get_pos_tags(document)
        tf_idf_nums = FeatExt.tf_idf_fun(document)
        embedded_document = []
        num_sents = len(document)
        if with_ner == True:
            for i in range(num_sents):
                num_words = len(document[i])
                embedded_sentence = []
                for j in range(num_words):
                    embedded_word = np.concatenate((doc_embeds[i][j],
                    np.array([pos_indices[i][j],tf_idf_nums[i][j],
                    doc_ner_tags[0][i][j],doc_ner_tags[1][i][j]])))
                    embedded_sentence.append(embedded_word)
                embedded_document.append(embedded_sentence)
        else:
            for i in range(num_sents):
                num_words = len(document[i])
                embedded_sentence = []
                for j in range(num_words):
                    embedded_word = np.concatenate((doc_embeds[i][j], np.array(([pos_indices[i][j],tf_idf_nums[i][j]]))))
                    embedded_sentence.append(embedded_word)
                embedded_document.append(embedded_sentence)
        return embedded_document

    def q_concat(question, with_ner = False, question_ner_tags = None):
        word_embeds = WordEmbed.glove_embed_sent(question)
        pos_indices = FeatExt.question_pos(question)
        num_words = len(question)
        embedded_question = []
        if with_ner == True:
            for j in range(num_words):
                embedded_word = np.concatenate((word_embeds[j],
                np.array([pos_indices[j],question_ner_tags[0][j],question_ner_tags[1][j]])))
                embedded_question.append(embedded_word)
        else:
            for j in range(num_words):
                embedded_word = np.concatenate((word_embeds[j], np.array([pos_indices[j]])))
                embedded_question.append(embedded_word)
        return embedded_question
