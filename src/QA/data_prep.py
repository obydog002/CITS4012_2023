import pandas as pd
import re

class DataPrep:
    def parse_tsv(path):
        df = pd.read_csv(path, sep="\t")
        return df

    def convert_pd_to_json(df):
        question_id_list = df.loc[:, "QuestionID"]
        question_id_list = list(set(question_id_list))
        question_id_list.sort()
        question_id = {}
        for i in df.index:  
            row = df.loc[i]
            id = row["QuestionID"]
            if id not in question_id:
                question_id[id] = {"document": []}
                question_id[id]["question"] = row["Question"]

            question_id[id]["document"].append((row["Sentence"], row["Label"]))
        return question_id

    def tokenize_json(json):
        qs = []
        docs = []
        ans = []
        for key in json.keys():
            toks = DataPrep.tokenize_question_and_doc(json[key])
            qs.append(toks[0])
            docs.append(toks[1])
            ans.append(toks[2])
        return qs, docs, ans

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
