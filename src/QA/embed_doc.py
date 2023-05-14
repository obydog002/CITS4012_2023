import numpy as np
from word_embed import WordEmbed
from feat_extract import FeatExt
from data_prep import DataPrep

# concatenate the embed and feature tags together
# could add future functionality to choose to exclude certain features for ablation

class EmbedAndConcat:
    def get_unrolled_embeddings(converted_json):
        q_inputs = []
        doc_inputs = []
        doc_targets = []
        for key in converted_json.keys():
            toks = DataPrep.tokenize_question_and_doc(converted_json[key])
            q_embeds = EmbedAndConcat.q_concat(toks[0])
            doc_embeds = EmbedAndConcat.doc_concat(toks[1])

            unrolled_doc_embeds = [word_embed for sentence in doc_embeds for word_embed in sentence]
            unrolled_doc_targets = [target for sentence in toks[2] for target in sentence]
            q_inputs.append(q_embeds)
            doc_inputs.append(unrolled_doc_embeds)
            doc_targets.append(unrolled_doc_targets)
        return q_inputs, doc_inputs, doc_targets

    def doc_concat(document, with_ner = False, doc_ner_tags = None, q_lemmas=[]):
        # default will run without NER, and return word match of all zeros
        doc_embeds = WordEmbed.glove_embed(document)
        pos_indices = FeatExt.get_pos_tags(document)
        tf_idf_nums = FeatExt.tf_idf_fun(document)
        word_matches = FeatExt.word_match(document, q_lemmas)
        embedded_document = []
        num_sents = len(document)
        if with_ner == True:
            for i in range(num_sents):
                num_words = len(document[i])
                for j in range(num_words):
                    embedded_word = np.concatenate((doc_embeds[i][j],
                    np.array([pos_indices[i][j],tf_idf_nums[i][j], word_matches[i][j],
                    doc_ner_tags[0][i][j],doc_ner_tags[1][i][j]])))
                    embedded_document.append(embedded_word)
        else:
            for i in range(num_sents):
                num_words = len(document[i])
                for j in range(num_words):
                    embedded_word = np.concatenate((doc_embeds[i][j], np.array(([pos_indices[i][j],tf_idf_nums[i][j]]))))
                    embedded_document.append(embedded_word)
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
