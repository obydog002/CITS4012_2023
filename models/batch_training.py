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

def load_and_get_tensors(q_cut_size="Max", doc_cut_size="Max", batch_size=128):
    # expensive in memory. So put in a function
    df = DataPrep.parse_tsv('../WikiQA-train.tsv')
    question_doc_raw_train = DataPrep.convert_pd_to_json(df)
    train_q_inputs, train_doc_inputs, train_doc_targets = EmbedAndConcat.get_unrolled_embeddings(question_doc_raw_train)

    df_test = DataPrep.parse_tsv('../WikiQA-test.tsv')
    question_doc_raw_test = DataPrep.convert_pd_to_json(df_test)
    test_q_inputs, test_doc_inputs, test_doc_targets = EmbedAndConcat.get_unrolled_embeddings(question_doc_raw_test)

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

    target2int = {"OOA": 0, "IOA": 1, "BOA": 1, "EOA": 1}
    int2target = {0: "OOA", 1: "IOA", 2: "BOA", 3: "EOA"}
    Pad.convert_targets(train_doc_targets, target2int)
    Pad.convert_targets(test_doc_targets, target2int)
    training_class_weights = StatHelper.get_class_weights(train_doc_targets, 2)

    train_dataset = TensorDataset(torch.Tensor(train_q_inputs), torch.Tensor(train_doc_inputs), torch.LongTensor(train_doc_targets))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

    test_question_tensor = torch.Tensor(test_q_inputs)
    test_doc_tensor = torch.Tensor(test_doc_inputs)
    test_target_tensor = torch.LongTensor(test_doc_targets)

    return train_loader, training_class_weights, test_question_tensor, test_doc_tensor, test_target_tensor

def do_training_and_eval(hidden_size = 100):
    train_loader, training_class_weights, test_question_tensor, test_doc_tensor, test_target_tensor = load_and_get_tensors()

    q_embed_size = list(train_loader)[0][0].shape[2]
    doc_embed_size = list(train_loader)[0][1].shape[2]
    class_size = 2
    question_rnn_model = QA_RNN.QuestionModel(q_embed_size, hidden_size).to(device)
    doc_rnn_model = QA_RNN.DocumentModel(doc_embed_size, hidden_size, class_size).to(device)

    n_iters = 5
    learning_rate=0.01
    criterion = nn.NLLLoss(weight=torch.Tensor(training_class_weights))
    question_model_optimizer = optim.SGD(question_rnn_model.parameters(), lr=learning_rate)
    document_model_optimizer = optim.SGD(doc_rnn_model.parameters(), lr=learning_rate)

    Trainer.trainIters(question_rnn_model, doc_rnn_model, n_iters, train_loader, criterion, question_model_optimizer, document_model_optimizer)
    Eval.evaluate(test_question_tensor, test_doc_tensor, test_target_tensor, question_rnn_model, doc_rnn_model)

do_training_and_eval()