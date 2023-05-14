import time
import math
import torch

# Helper functions for training
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_question_tensor, input_document_tensor, target_tensor, question_model, document_model, question_optimizer, document_optimizer, criterion):
    loss = 0    

    question_optimizer.zero_grad()
    document_optimizer.zero_grad()
    # get output from question model
    question_output = question_model(input_question_tensor)

    # predictions from the document model with output from question for attention
    document_output = document_model(input_document_tensor, question_output)
    
    loss = criterion(document_output.view(-1, document_output.shape[-1]), target_tensor.view(-1))
    loss.backward()
    
    question_optimizer.step()
    document_optimizer.step()

    return loss.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def trainIters(question_model, document_model, n_iters, train_loader, criterion, question_model_optimizer, document_model_optimizer, print_every=5, plot_every=10):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        
        for iter in range(1, n_iters):
            for question_input,doc_input,targets in train_loader:
                question_input = question_input.to(device)
                doc_input = doc_input.to(device)
                targets = targets.to(device)

                loss = train(question_input, doc_input, targets, question_model, document_model, question_model_optimizer, document_model_optimizer, criterion)
                print_loss_total += loss
                plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0