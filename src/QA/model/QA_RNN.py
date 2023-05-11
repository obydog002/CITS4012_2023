import torch.nn as nn
import torch.nn.functional as F
import torch

# QA with document RNN model
# use attention between the question and document.
class DocumentModel(nn.Module):
    ATTN_TYPE_DOT_PRODUCT = "Dot Product"
    ATTN_TYPE_SCALE_DOT_PRODUCT = "Scale Dot Product"

    def __init__(self, n_input, n_hidden, n_class):
        super(DocumentModel, self).__init__()

        self.rnn = nn.RNN(n_input, n_hidden, batch_first=True, bidirectional=True)
        self.rnn_to_att = nn.Linear(2*n_hidden,n_hidden)
        self.out = nn.Linear(2*n_hidden,n_class)

    def dot(self, t1, t2):
        batch_size = t1.shape[0]
        layer_size = t1.shape[1]
        print(t1.shape)
        print(t2.shape)
        return torch.bmm(t1.view(batch_size, 1, layer_size), t2.view(batch_size, layer_size, 1))

    def calc_attention(self, hidden, question_hidden, method):
        if method == DocumentModel.ATTN_TYPE_DOT_PRODUCT:
            weights =  F.softmax(self.dot(hidden, question_hidden), dim=-1)
            attention_output = self.dot(weights, question_hidden.unsqueeze(0))
            catted_output = torch.cat((attention_output, hidden), 1)
            
        return catted_output

    def forward(self, input, question_hidden):        
        _, h_n = self.rnn(input)
        # concat the last hidden states for both directions
        rnn_out = torch.cat((h_n[-1,:,:],h_n[-2,:,:]),1)
        hidden_out = self.rnn_to_att(rnn_out)

        catted_output = self.calc_attention(hidden_out, question_hidden, DocumentModel.ATTN_TYPE_DOT_PRODUCT)

        output = F.softmax(self.out(catted_output), dim=1)
        return output, hidden_out

class QuestionModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_hidden_out):
        super(QuestionModel, self).__init__()

        self.rnn = nn.RNN(n_input, n_hidden, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*n_hidden, n_hidden_out)

    def forward(self, input):
        _, h_n = self.rnn(input)

        hidden_out = torch.cat((h_n[0,:,:],h_n[1,:,:]),1)
        output = self.linear(hidden_out)
        return output


