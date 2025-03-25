import torch, os

os.environ['HF_HOME'] = './cache'
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset

from train import evaluate, collate

DATA_PATH = "../data/test"

if __name__ == '__main__':
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # Load the model
    model = RobertaForSequenceClassification.from_pretrained("models/CodeBERT-solidifi").to('cuda')
    # Load the dataset
    if os.path.exists(DATA_PATH + "/dataset.pt"):
        dataset = torch.load(DATA_PATH + "/dataset.pt", weights_only=False)
    else:
        dataset = SmardityDataset(DATA_PATH, tokenizer)
        torch.save(dataset, DATA_PATH + "/dataset.pt")
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate)
    # Evaluate the model
    evaluate(model, dataloader, 'cuda')
