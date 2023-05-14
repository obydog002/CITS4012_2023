import torch.nn as nn
import torch.nn.functional as F
import torch

# QA with document RNN model
# use attention between the question and document.
class DocumentModel(nn.Module):
    ATTN_TYPE_DOT_PRODUCT = "Dot Product"
    ATTN_TYPE_SCALE_DOT_PRODUCT = "Scale Dot Product"

    def __init__(self, n_input, rnn_hidden, n_class):
        super(DocumentModel, self).__init__()

        self.rnn = nn.RNN(n_input, rnn_hidden, batch_first=True, bidirectional=True)
        # 2* for bidrectional rnn
        self.out = nn.Linear(2*rnn_hidden, n_class)

    def calc_attention(self, hidden, question_hidden, method):
        if method == DocumentModel.ATTN_TYPE_DOT_PRODUCT:
            weights =  F.softmax(torch.bmm(hidden, question_hidden.transpose(1,2)), dim=-1)
            attention_output = torch.bmm(weights, question_hidden)
            
        return attention_output

    def forward(self, input, question_hidden):        
        rnn_output, _ = self.rnn(input)

        attention_output = self.calc_attention(rnn_output, question_hidden, DocumentModel.ATTN_TYPE_DOT_PRODUCT)

        output = F.log_softmax(self.out(attention_output), dim=-1)
        return output

class QuestionModel(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(QuestionModel, self).__init__()

        self.rnn = nn.RNN(n_input, n_hidden, batch_first=True, bidirectional=True)

    def forward(self, input):
        rnn_output, _ = self.rnn(input)

        return rnn_output
