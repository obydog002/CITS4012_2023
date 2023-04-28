import pandas as pd

class DataPrep:
    def parse_tsv(path):
        df = pd.read_csv(path, sep="\t")

        return df

    def parse_QA(question, document, label):
        print(question)
        print(document)
        print(label)

    def tokenizer(words):
        pass

    def encode_answer_types(data):
        pass