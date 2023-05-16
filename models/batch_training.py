import sys
sys.path.append('../src/QA')

from word_embed import WordEmbed
from data_prep import DataPrep
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from feat_extract import FeatExt
from embed_doc import EmbedAndConcat
from stat_helper import StatHelper
from pad import Pad
from model import QA_RNN
from model.train import Trainer
from model.eval import Eval

Out_And_In_target2int = {"OOA": 0, "IOA": 1, "BOA": 1, "EOA": 1}
Out_In_Beg_End_target2int = {"OOA": 0, "IOA": 1, "BOA": 2, "EOA": 3}
Bef_In_Aft_target2int = {"BA": 0, "IA": 1, "AA": 2}

question_doc_raw_train = DataPrep.convert_pd_to_json(DataPrep.parse_tsv("../WikiQA-train.tsv"))
question_doc_raw_test = DataPrep.convert_pd_to_json(DataPrep.parse_tsv("../WikiQA-test.tsv"))

def load_and_get_tensors(q_cut_size="Max", doc_cut_size="Max", answer_type="Out_And_In", befaft = False, doc_with_pos = True, doc_with_tfidf = True, doc_with_ner = False, doc_with_wm = False, q_with_pos = True, q_with_ner = False):
    train_q_inputs, train_doc_inputs, train_doc_targets = EmbedAndConcat.get_unrolled_embeddings(question_doc_raw_train, befaft, doc_with_pos, doc_with_tfidf, doc_with_ner, doc_with_wm, q_with_pos, q_with_ner)

    test_q_inputs, test_doc_inputs, test_doc_targets = EmbedAndConcat.get_unrolled_embeddings(question_doc_raw_test, befaft, doc_with_pos, doc_with_tfidf, doc_with_ner, doc_with_wm, q_with_pos, q_with_ner)

    if q_cut_size == "Max":
        q_cut_size = max(Pad.get_max(train_q_inputs), Pad.get_max(test_q_inputs))
    if doc_cut_size == "Max":
        doc_cut_size = max(Pad.get_max(train_doc_inputs), Pad.get_max(test_doc_inputs))

    Pad.cut_pad_to(q_cut_size, train_q_inputs)
    Pad.cut_pad_to(q_cut_size, test_q_inputs)
    Pad.cut_pad_to(doc_cut_size, train_doc_inputs)
    Pad.cut_pad_to(doc_cut_size, test_doc_inputs)
    Pad.cut_pad_to(doc_cut_size, train_doc_targets, target=True)
    Pad.cut_pad_to(doc_cut_size, test_doc_targets, target=True)

    if answer_type == "Out_And_In":
        target2int = Out_And_In_target2int
    elif answer_type == "Out_In_Beg_End":
        target2int = Out_In_Beg_End_target2int
    else:
        target2int = Bef_In_Aft_target2int

    Pad.convert_targets(train_doc_targets, target2int)
    Pad.convert_targets(test_doc_targets, target2int)
    number_of_classes = max(target2int.values()) + 1

    print(number_of_classes)
    training_class_weights = StatHelper.get_class_weights(train_doc_targets, number_of_classes)


    train_question_tensor = torch.Tensor(train_q_inputs)
    train_doc_tensor = torch.Tensor(train_doc_inputs)
    train_target_tensor = torch.Tensor(train_doc_targets)

    test_question_tensor = torch.Tensor(test_q_inputs)
    test_doc_tensor = torch.Tensor(test_doc_inputs)
    test_target_tensor = torch.LongTensor(test_doc_targets)

    return train_question_tensor, train_doc_tensor, train_target_tensor, training_class_weights, test_question_tensor, test_doc_tensor, test_target_tensor, number_of_classes

def do_training_and_eval(train_question_tensor, train_doc_tensor, train_target_tensor, train_loader, test_question_tensor, test_doc_tensor, test_target_tensor, training_class_weights, hidden_size, number_of_classes, learning_rate):
    q_embed_size = list(train_loader)[0][0].shape[2]
    doc_embed_size = list(train_loader)[0][1].shape[2]
    question_rnn_model = QA_RNN.QuestionModel(q_embed_size, hidden_size).to(device)
    doc_rnn_model = QA_RNN.DocumentModel(doc_embed_size, hidden_size, number_of_classes).to(device)

    criterion = nn.NLLLoss(weight=torch.Tensor(training_class_weights))
    question_model_optimizer = optim.SGD(question_rnn_model.parameters(), lr=learning_rate)
    document_model_optimizer = optim.SGD(doc_rnn_model.parameters(), lr=learning_rate)
    
    print("Starting training..")
    # train model for 1, 5, 10, 20, 40 iterations, and record performances
    iters_inc = [1,4,5,10,20] 
    total_iters = 0
    for inc in iters_inc:
        #Trainer.trainIters(question_rnn_model, doc_rnn_model, inc, train_loader, criterion, question_model_optimizer, document_model_optimizer)
        #Eval.evaluate(test_question_tensor, test_doc_tensor, test_target_tensor, question_rnn_model, doc_rnn_model)
        total_iters += inc
        print(f"trained for {total_iters}"....)

from itertools import product

def train_all_models_on_param_grid(loading_params, training_params):
    def get_unrolled_params(params):
        keys, values = zip(*params.items())
        return [dict(zip(keys, p)) for p in product(*values)]

    loading_params = get_unrolled_params(loading_params)
    training_params = get_unrolled_params(training_params)

    for loading_param in loading_params:

        train_question_tensor, train_doc_tensor, train_target_tensor, training_class_weights, test_question_tensor, test_doc_tensor, test_target_tensor, number_of_classes = load_and_get_tensors(doc_cut_size = loading_param["doc_cut_pad_to_length"], answer_type=loading_param["answer_type"], befaft=loading_param["befaft"], doc_with_pos=loading_param["doc_with_pos"], doc_with_tfidf=loading_param["doc_with_tfidf"], doc_with_ner=loading_param["doc_with_ner"], doc_with_wm=loading_param["doc_with_wm"], q_with_pos=loading_param["q_with_pos"], q_with_ner=loading_param["q_with_ner"])

        for batch_param in batch_params:
            train_dataset = TensorDataset(train_question_tensor, train_doc_tensor, train_target_tensor)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_param, shuffle=True) 

            for training_param in training_params:
                do_training_and_eval(train_question_tensor, train_doc_tensor, train_target_tensor, train_loader, test_question_tensor, test_doc_tensor, test_target_tensor, training_class_weights, 100, number_of_classes, training_param["learning_rate"])


loading_params = {"doc_cut_pad_to_length": [256, "Max"], 
                  "answer_type": ["Out_And_In", "Out_In_Beg_End", "Bef_In_Aft"],
                  "befaft": [False, True], "doc_with_pos": [False, True], "doc_with_tfidf": [False, True], "doc_with_ner": [False, True], "doc_with_wm": [False, True], "q_with_pos": [False, True], "q_with_ner": [False, True]}
batch_params =[32, 128, 256]
training_params = {"learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1]}

train_all_models_on_param_grid(loading_params, training_params)
