import os, torch
os.environ['HF_HOME'] = './cache'

from transformers import RobertaTokenizer, RobertaModel
from process_data import SmardityDataset

# Load model directly
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

def train_model(model, tokenizer, dataset):
    '''
    Fine tune the model on the dataset
    '''
    # Setup the model parameters
    # 1. Freeze the pre-trained part of model 
    for param in model.parameters():
        param.requires_grad = False
    # 2. Add classification head
    classifier = torch.nn.Linear(model.config.hidden_size, len(dataset.labels.keys()))
    model = torch.nn.Sequential(model, classifier).cuda()

    # Set running metadata
    # 1. Device
    # 2. Loss function
    # 3. Optimizer
    # 4. Scheduler

    # Set model hyperparameters
    # 1. Batch size
    # 2. Learning rate
    # 3. Number of epochs
    # 4. Keep track of loss and best model

    # Train the model    


def evaluate(model, dataset):
    '''
    Evaluate the model on the dataset
    '''
    pass 


if __name__ == "__main__":
    dataset = SmardityDataset("data/contracts")
    print(f"Dataset length: {len(dataset)}")