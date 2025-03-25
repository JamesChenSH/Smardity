import torch, os

os.environ['HF_HOME'] = './cache'
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset

from train import evaluate, collate

DATA_PATH = "data/test"    # TODO: may need to update this to match your local env
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if __name__ == '__main__':
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # Load the model
    model = RobertaForSequenceClassification.from_pretrained("models/CodeBERT-solidifi_finetuned").to(DEVICE)    # TODO: use fine-tuned model instead
    # Load the dataset
    if os.path.exists(DATA_PATH + "/dataset.pt"):
        dataset = torch.load(DATA_PATH + "/dataset.pt", weights_only=False)
    else:
        dataset = SmardityDataset(DATA_PATH, tokenizer)
        torch.save(dataset, DATA_PATH + "/dataset.pt")
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate)
    # Evaluate the model
    evaluate(model, dataloader, DEVICE)
