import time
import math
import torch

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
    def trainIters(question_model, document_model, n_iters, train_loader, criterion, question_model_optimizer, document_model_optimizer):
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
