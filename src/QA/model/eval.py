from sklearn.metrics import classification_report
import torch

class Eval:
    def evaluate(input_question_tensor, input_document_tensor, target_tensor, question_model, document_model, criterion):
        with torch.no_grad():
            question_output = question_model(input_question_tensor)
            
            document_output = document_model(input_document_tensor, question_output)
            predicts = torch.argmax(document_output.view(-1, document_output.shape[-1]), dim=-1)

            loss = criterion(document_output.view(-1, document_output.shape[-1]), target_tensor.view(-1)) 
            return loss, classification_report(predicts, target_tensor.view(-1), output_dict = True)
