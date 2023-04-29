import pandas as pd

class DataPrep:
    def parse_tsv(path):
        df = pd.read_csv(path, sep="\t")
        return df

    def tokenize_question_and_doc(question_doc_list):
        tok_q = DataPrep.tokenize_question(question_doc_list["question"])
        tok_doc = []
        for doc_tup in question_doc_list["document"]:
            tok_doc.append(DataPrep.tokenize_doc(doc_tup))
        return (tok_q, tok_doc)

    def tokenize_question(question):
        sentence = str.split(question, " ")
        return sentence

    def tokenize_doc(doc_tup):
        sentence = str.split(doc_tup[0], " ")
        label = doc_tup[1]
        if label == 0: # non-answer sentence
            return [[word, "OOA"] for word in sentence]
        if label  == 1: # answer
            encoded = [[word, "IOA"] for word in sentence]
            encoded[0][1] = "BOA"
            encoded[-1][1] = "EOA"
            return encoded