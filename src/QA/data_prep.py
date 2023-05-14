import pandas as pd
import re

class DataPrep:
    def parse_tsv(path):
        df = pd.read_csv(path, sep="\t")
        return df

    def tokenize_question_and_doc(question_doc_list, befaft=False):
        tok_q = DataPrep.tokenize_question(question_doc_list["question"])
        tok_doc = []
        tok_ans = []
        if befaft == True: # before-answer-after labeling
            # labels words with "before answer", "in answer", and "after answer"
            tok_q = DataPrep.tokenize_question(question_doc_list["question"])
            tok_doc = []
            tok_ans = []
            ans_lab = 'BA' # intialise with before answer token
            for doc_tup in question_doc_list["document"]:
                if doc_tup[1] == 1:
                    ans_lab = "IA" # answer
                sentence_toks = re.sub('(?<=[^ ])(?=[.,!?()\'\"\-:;])|(?<=[.,!?()\'\"\-:;])(?=[^ ])|\s{2,}', r' ', doc_tup[0]).lower().split()
                doc = []
                answer = []
                for word in sentence_toks:
                    doc.append(word)
                    answer.append(ans_lab)
                tok_doc.append(doc)
                tok_ans.append(answer)
                if doc_tup[1] == 1:
                    ans_lab = 'AA' # change label to after answer
        else: # outside-answer begining-of-answer inside-of-answer end-of-answer labeling
            for doc_tup in question_doc_list["document"]:
                tok_tup = DataPrep.tokenize_doc(doc_tup)
                tok_doc.append(tok_tup[0])
                tok_ans.append(tok_tup[1])
        return (tok_q, tok_doc, tok_ans)

    def tokenize_question(question):
        # this regex finds all .,!?()'"-:; punctuation and adds a space before or after it, if needed
        #
        sentence_toks = re.sub('(?<=[^ ])(?=[.,!?()\'\"\-:;])|(?<=[.,!?()\'\"\-:;])(?=[^ ])|\s{2,}', r' ', question).lower().split()
        return sentence_toks

    def tokenize_doc(doc_tup):
        # this regex finds all .,!?()'"-:; punctuation and adds a space before or after it, if needed
        sentence_toks = re.sub('(?<=[^ ])(?=[.,!?()\'\"\-:;])|(?<=[.,!?()\'\"\-:;])(?=[^ ])|\s{2,}', r' ', doc_tup[0]).lower().split()
        label = doc_tup[1]
        answer = []
        doc = []
        if label == 0: # non-answer sentence
            for word in sentence_toks:
                doc.append(word)
                answer.append("OOA")
            return doc,answer
        if label  == 1: # answer
            for word in sentence_toks:
                doc.append(word)
                answer.append("IOA")
            answer[0] = "BOA"
            answer[-1] = "EOA"
            return (doc,answer)

    def tokenize_question_and_doc_befaft(question_doc_list):
        # labels words with "before answer", "in answer", and "after answer"
        tok_q = DataPrep.tokenize_question(question_doc_list["question"])
        tok_doc = []
        tok_ans = []
        ans_lab = 'BA' # intialise with before answer token
        for doc_tup in question_doc_list["document"]:
            if doc_tup[1] == 1:
                ans_lab = "IA" # answer
            sentence_toks = re.sub('(?<=[^ ])(?=[.,!?()\'\"\-:;])|(?<=[.,!?()\'\"\-:;])(?=[^ ])|\s{2,}', r' ', doc_tup[0]).lower().split()
            doc = []
            answer = []
            for word in sentence_toks:
                doc.append(word)
                answer.append(ans_lab)
            tok_doc.append(doc)
            tok_ans.append(answer)
            if doc_tup[1] == 1:
                ans_lab = 'AA' # change label to after answer
        return (tok_q, tok_doc, tok_ans)
