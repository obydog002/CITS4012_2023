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

    def doc_concat(document, with_pos = True, with_tfidf = True, with_ner = False, doc_ner_tags = None, with_wm = False, q_lemmas = None):
        # concatenates embeddings depending on arguments provided  
        doc_embeds = WordEmbed.glove_embed(document)
        num_sents = len(document)
        num_words = []
        embedded_document = []
        for i in range(num_sents):
            num_words.append(len(document[i]))
        if with_pos == True:
            pos_indices = FeatExt.get_pos_tags(document)
            for i in range(num_sents):
                for j in range(num_words[i]):
                    doc_embeds[i][j] = np.concatenate((doc_embeds[i][j], np.array([pos_indices[i][j]])))
        if with_tfidf == True:
            tf_idf_nums = FeatExt.tf_idf_fun(document)
            for i in range(num_sents):
                for j in range(num_words[i]):
                    doc_embeds[i][j] = np.concatenate((doc_embeds[i][j], np.array([tf_idf_nums[i][j]])))
        if with_wm == True:
            word_matches = FeatExt.word_match(document, q_lemmas)
            for i in range(num_sents):
                for j in range(num_words[i]):
                    doc_embeds[i][j] = np.concatenate((doc_embeds[i][j], np.array([word_matches[i][j]])))
        if with_ner == True:
            for i in range(num_sents):
                for j in range(num_words[i]):
                    doc_embeds[i][j] = np.concatenate((doc_embeds[i][j], np.array([doc_ner_tags[0][i][j],doc_ner_tags[1][i][j]])))
        for i in range(num_sents):
            for j in range(num_words[i]):
                embedded_document.append(doc_embeds[i][j]) # add embedded word to document
        return embedded_document

    def q_concat(question, with_pos = True, with_ner = False, question_ner_tags = None):
        word_embeds = WordEmbed.glove_embed_sent(question)
        num_words = len(question)
        embedded_question = []
        if with_pos == True:
            pos_indices = FeatExt.question_pos(question)
            for j in range(num_words):
                word_embeds[j] = np.concatenate((word_embeds[j], np.array([pos_indices[j]])))
        if with_ner == True:
            for j in range(num_words):
                word_embeds[j] = np.concatenate((word_embeds[j], np.array([question_ner_tags[0][j],question_ner_tags[1][j]])))
        for j in range(num_words):
            embedded_question.append(word_embeds[j])
        return embedded_question
