import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

cos = nn.CosineSimilarity(dim=2)

# QA with document RNN model
# use attention between the question and document.
class DocumentModel(nn.Module):
    ATTN_TYPE_DOT_PRODUCT = "Dot Product"
    ATTN_TYPE_SCALED_DOT_PRODUCT = "Scaled Dot Product"
    ATTN_TYPE_COSINE = "Cosine"

    HIDDEN_TYPE_RNN = "RNN"
    HIDDEN_TYPE_LSTM = "LSTM"
    HIDDEN_TYPE_GRU = "GRU"

    def __init__(self, n_input, n_hidden, n_class, attention_type=ATTN_TYPE_DOT_PRODUCT, hidden_layers=1, hidden_layer=HIDDEN_TYPE_RNN, bidirectional=True):
        super(DocumentModel, self).__init__()

        self.rnn = nn.RNN(n_input, n_hidden, hidden_layers, batch_first=True, bidirectional=bidirectional)
        linear_layer_size = n_hidden
        if bidirectional:
            linear_layer_size = 2*linear_layer_size
        self.out = nn.Linear(linear_layer_size, n_class)

    def calc_attention(self, hidden, question_hidden, method):
        if method == DocumentModel.ATTN_TYPE_DOT_PRODUCT:
            weights =  F.softmax(torch.bmm(hidden, question_hidden.transpose(1,2)), dim=-1)
            attention_output = torch.bmm(weights, question_hidden)
        elif method == DocumentModel.ATTN_TYPE_SCALED_DOT_PRODUCT:
            weights = F.softmax(torch.bmm(hidden, question_hidden.transpose(1,2))/np.sqrt(hidden.shape[2]), dim=-1)
            attention_output = torch.bmm(weights, question_hidden)
        elif method == DocumentModel.ATTN_TYPE_COSINE:
            weights = F.softmax(cos(hidden, question_hidden.transpose(1,2)), dim=-1)
            attention_output = torch.bmm(weights, question_hidden)
        return attention_output

    def forward(self, input, question_hidden):        
        rnn_output, _ = self.rnn(input)

        attention_output = self.calc_attention(rnn_output, question_hidden, attention_type)

        # log softmax as we use a negative log likelihood loss
        output = F.log_softmax(self.out(attention_output), dim=-1)
        return output

class QuestionModel(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(QuestionModel, self).__init__()

        self.rnn = nn.RNN(n_input, n_hidden, batch_first=True, bidirectional=True)

    def forward(self, input):
        rnn_output, _ = self.rnn(input)

        return rnn_output
