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

import time

Out_And_In_target2int = {"OOA": 0, "IOA": 1, "BOA": 1, "EOA": 1}
Out_In_Beg_End_target2int = {"OOA": 0, "IOA": 1, "BOA": 2, "EOA": 3}
Bef_In_Aft_target2int = {"BA": 0, "IA": 1, "AA": 2}

question_doc_raw_train = DataPrep.convert_pd_to_json(DataPrep.parse_tsv("../WikiQA-train.tsv"))
question_doc_raw_test = DataPrep.convert_pd_to_json(DataPrep.parse_tsv("../WikiQA-test.tsv"))

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(filename='results.log', mode="a", encoding='utf-8', delay=False)

logger.addHandler(fh)

def load_and_get_tensors(loading_param):
    q_cut_size = loading_param["q_cut_size"]
    doc_cut_size = loading_param["doc_cut_size"]
    answer_type = loading_param["answer_type"]
    befaft = loading_param["befaft"]
    doc_with_pos = loading_param["doc_with_pos"]
    doc_with_tfidf = loading_param["doc_with_tfidf"]
    doc_with_ner = loading_param["doc_with_ner"]
    doc_with_wm = loading_param["doc_with_wm"]
    q_with_pos = loading_param["q_with_pos"]
    q_with_ner = loading_param["q_with_ner"]
    train_q_inputs, train_doc_inputs, train_doc_targets = EmbedAndConcat.get_unrolled_embeddings(question_doc_raw_train, befaft, doc_with_pos, doc_with_tfidf, doc_with_ner, doc_with_wm, q_with_pos, q_with_ner)

    test_q_inputs, test_doc_inputs, test_doc_targets = EmbedAndConcat.get_unrolled_embeddings(question_doc_raw_test, befaft, doc_with_pos, doc_with_tfidf, doc_with_ner, doc_with_wm, q_with_pos, q_with_ner)

    if q_cut_size == "Max":
        q_cut_size = max(Pad.get_max(train_q_inputs), Pad.get_max(test_q_inputs))
    if doc_cut_size == "Max":
        doc_cut_size = max(Pad.get_max(train_doc_inputs), Pad.get_max(test_doc_inputs))

    # ignore answer_type if we are doing befaft
    if befaft:
        target2int = Bef_In_Aft_target2int
    elif answer_type == "Out_And_In":
        target2int = Out_And_In_target2int
    elif answer_type == "Out_In_Beg_End":
        target2int = Out_In_Beg_End_target2int

    Pad.cut_pad_to(q_cut_size, train_q_inputs)
    Pad.cut_pad_to(q_cut_size, test_q_inputs)
    Pad.cut_pad_to(doc_cut_size, train_doc_inputs)
    Pad.cut_pad_to(doc_cut_size, test_doc_inputs)
    Pad.cut_pad_to(doc_cut_size, train_doc_targets, target=True)
    Pad.cut_pad_to(doc_cut_size, test_doc_targets, target=True)

    Pad.convert_targets(train_doc_targets, target2int)
    Pad.convert_targets(test_doc_targets, target2int)
    number_of_classes = max(target2int.values()) + 1

    training_class_weights = StatHelper.get_class_weights(train_doc_targets, number_of_classes)

    train_question_tensor = torch.Tensor(train_q_inputs)
    train_doc_tensor = torch.Tensor(train_doc_inputs)
    train_target_tensor = torch.LongTensor(train_doc_targets)

    test_question_tensor = torch.Tensor(test_q_inputs)
    test_doc_tensor = torch.Tensor(test_doc_inputs)
    test_target_tensor = torch.LongTensor(test_doc_targets)

    return train_question_tensor, train_doc_tensor, train_target_tensor, training_class_weights, test_question_tensor, test_doc_tensor, test_target_tensor, number_of_classes

def do_training_and_eval(train_question_tensor, train_doc_tensor, train_target_tensor, train_loader, test_question_tensor, test_doc_tensor, test_target_tensor, training_class_weights, number_of_classes, training_param):
    learning_rate = training_param["learning_rate"]
    bidirectional = training_param["bidirectional"]
    attention_type = training_param["attention_type"]
    hidden_type = training_param["hidden_type"]
    doc_hidden_layers = training_param["doc_hidden_layers"]
    hidden_size = training_param["hidden_size"]
    iters_inc = training_param["iters_inc"]

    q_embed_size = list(train_loader)[0][0].shape[2]
    doc_embed_size = list(train_loader)[0][1].shape[2]

    doc_rnn_model = QA_RNN.DocumentModel(doc_embed_size, hidden_size, number_of_classes, hidden_layers = doc_hidden_layers, bidirectional=bidirectional).to(device)
    question_rnn_model = QA_RNN.QuestionModel(q_embed_size, hidden_size, hidden_layers = doc_hidden_layers, bidirectional=bidirectional).to(device)

    criterion = nn.NLLLoss(weight=torch.Tensor(training_class_weights))
    question_model_optimizer = optim.SGD(question_rnn_model.parameters(), lr=learning_rate)
    document_model_optimizer = optim.SGD(doc_rnn_model.parameters(), lr=learning_rate)

    results = {}
    total_iters = 0
    for inc in iters_inc:
        Trainer.trainIters(question_rnn_model, doc_rnn_model, inc, train_loader, criterion, question_model_optimizer, document_model_optimizer)
        train_loss, train_report = Eval.evaluate(train_question_tensor, train_doc_tensor, train_target_tensor, question_rnn_model, doc_rnn_model, criterion)
        test_loss, test_report = Eval.evaluate(test_question_tensor, test_doc_tensor, test_target_tensor, question_rnn_model, doc_rnn_model, criterion)
        total_iters += inc
        results[total_iters] = {"train_loss": train_loss, "train_report": train_report, "test_loss": test_loss, "test_report": test_report}
        logger.info(f"Trained on {total_iters}.")

    return results

load_times = []
batch_times = []
train_times = []

master_results = {}
import pickle
import os

if os.path.exists("results.pkl"):
    with open('results.pkl', 'rb') as f:
        master_results = pickle.load(f)

def save_master_results():
    with open('results.pkl', 'wb') as f:
        pickle.dump(master_results, f)

import statistics
def get_expected_time_s(times, n_left):
    if len(times) == 0:
        return n_left * 60 # default 60s
    else:
        return statistics.fmean(times) * n_left

# catersian product of list of dicts
def cart(list_dict1, list_dict2):
    l = []
    for d1 in list_dict1:
        for d2 in list_dict2:
            l.append(d1 | d2)
    return l

# returns all matching params specified by presence of fields in param
def get_matching_params(param, all_params):
    matched = []
    for candidate in all_params:
        if param.items() <= candidate.items():
            matched.append(candidate)

    return matched
    
# checks in the master list, if the next set of params has already been covered in the master list
def check_if_training_needed(param, all_params):
    matched = get_matching_params(param, all_params)
    for m in matched:
        if frozenset(m.items()) not in master_results:
            return True

    return False

def print_time_nicely(secs):
    secs = int(secs)
    mins = 0
    hours = 0
    time_str = f"{secs % 60}s"
    if secs > 60:
        mins = secs // 60
        time_str = f"{mins % 60}m " + time_str
    if mins > 60:
        hours = mins // 60
        time_str = f"{hours}h " + time_str
    logger.info(f"Expected time {time_str}")

from itertools import product
def train_all_models_on_param_grid(loading_params, batch_params, training_params):
    def get_unrolled_params(params):
        keys, values = zip(*params.items())
        return [dict(zip(keys, p)) for p in product(*values)]

    loading_params = get_unrolled_params(loading_params)
    batch_params = get_unrolled_params(batch_params)
    training_params = get_unrolled_params(training_params)

    all_params = cart(cart(loading_params, batch_params), training_params)

    loading_params_len = len(loading_params)
    batch_params_len = len(batch_params)
    training_params_len = len(training_params)
    logger.info(f"total models: {loading_params_len*batch_params_len*training_params_len}")

    for li, loading_param in enumerate(loading_params):
        # check if training is neccessary
        if not check_if_training_needed(loading_param, all_params):
            logger.info("skipping at load.")
            continue

        logger.info("Starting load...")
        logger.info(f"load {li+1}/{loading_params_len}")
        before_time = time.time()
        train_question_tensor, train_doc_tensor, train_target_tensor, training_class_weights, test_question_tensor, test_doc_tensor, test_target_tensor, number_of_classes = load_and_get_tensors(loading_param)
        load_times.append(time.time() - before_time)

        for bi, batch_param in enumerate(batch_params):
            if not check_if_training_needed(loading_param | batch_param, all_params):
                logger.info("skipping at batch.")
                continue

            logger.info("Starting batch load...")
            batch_full_index = li*batch_params_len + bi
            logger.info(f"batch {batch_full_index + 1}/{batch_params_len*loading_params_len}")
            before_time = time.time()
            train_dataset = TensorDataset(train_question_tensor, train_doc_tensor, train_target_tensor)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_param["batch"], shuffle=True) 
            batch_times.append(time.time() - before_time)

            for ti, training_param in enumerate(training_params):
                logger.info(get_matching_params(loading_param | batch_param | training_param, all_params))
                if not check_if_training_needed(loading_param | batch_param | training_param, all_params):
                    logger.info("skipping at training.")
                    continue

                # freeze params for model description
                frozen_param = loading_param | batch_param | training_param
                frozen_param = frozenset(frozen_param.items()) 

                # skip computed values
                if frozen_param in master_results:
                    logger.info("second training skip")
                    continue

                train_full_index = training_params_len*batch_full_index + ti
                loading_s_to_expect = get_expected_time_s(load_times, loading_params_len - li - 1)
                batch_s_to_expect = get_expected_time_s(batch_times, loading_params_len*batch_params_len - batch_full_index - 1)
                train_s_to_expect = get_expected_time_s(train_times, loading_params_len*batch_params_len*training_params_len - train_full_index - 1)
                print_time_nicely(loading_s_to_expect + batch_s_to_expect + train_s_to_expect)

                logger.info("Starting training model...")
                logger.info(f"train {train_full_index + 1}/{batch_params_len*loading_params_len*training_params_len}")
                logger.info(frozen_param)

                before_time = time.time()
                results = do_training_and_eval(train_question_tensor, train_doc_tensor, train_target_tensor, train_loader, test_question_tensor, test_doc_tensor, test_target_tensor, training_class_weights, number_of_classes, training_param)
                train_times.append(time.time() - before_time)

                master_results[frozen_param] = results
                save_master_results()


loading_params = {"q_cut_size": ["Max"],
                  "doc_cut_size": [256], 
                  "answer_type": ["Out_And_In"],
                  "befaft": [False], "doc_with_pos": [False], "doc_with_tfidf": [True], 
                  "doc_with_ner": [True], "doc_with_wm": [False], "q_with_pos": [False], 
                  "q_with_ner": [True]}
batch_params = {"batch": [128]}
training_params = {"learning_rate": [0.1, 0.5, 1.0], "bidirectional": [False], 
        "attention_type": [QA_RNN.DocumentModel.ATTN_TYPE_DOT_PRODUCT],
        "hidden_type": [QA_RNN.DocumentModel.HIDDEN_TYPE_RNN, QA_RNN.DocumentModel.HIDDEN_TYPE_LSTM, QA_RNN.DocumentModel.HIDDEN_TYPE_GRU],
        "doc_hidden_layers": [1,2,3,4,5],
        "hidden_size": [100],
        "iters_inc": [(1,4,5,10,20,40)]}

train_all_models_on_param_grid(loading_params, batch_params, training_params)
