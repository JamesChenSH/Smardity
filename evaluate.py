import torch, os

os.environ['HF_HOME'] = './cache'
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset, collate
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

DATA_PATH = "./data/test"    # TODO: may need to update this to match your local env
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def evaluate(model, dataloader: torch.utils.data.dataloader.DataLoader, label_dict=None, device="cuda"):
    '''
    Evaluate the model on the dataset
    '''
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    
    for ids, attention_mask, labels in dataloader:
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        y_true.extend(labels.cpu().numpy().tolist())
        with torch.no_grad(), torch.autocast(device):
            outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labels)
            loss_val = outputs[0]
        y_pred.extend(outputs[1].argmax(dim=1).cpu().numpy().tolist())
        losses.append(loss_val.item())
    
    # Compute average loss
    loss_val = sum(losses) / len(losses)
    # Overall metrics
    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    # Compute the average loss
    loss_val = torch.tensor(losses).mean()
    # Log the metrics
    print(f"Loss: {loss_val}, Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    # Test time only metrics when label_dict is given
    if label_dict is not None:
        # Get class names
        val_to_class = {v: k for k, v in label_dict.items()}
        class_names = [val_to_class[i] for i in range(len(list(val_to_class.keys())))]
        report = classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
        
        # Lists to store metrics
        accs = []
        precisions = []
        recalls = []
        f1s = []
        classes = []

        # computing per class metrics
        for i in range(len(class_names)):
            classes.append(val_to_class[i])
            
            # Get indices where y_true equals the current class i
            true_indices = [idx for idx, y in enumerate(y_true) if y == i]
            
            y_true_class = [1 for idx in true_indices]
            y_pred_class = [1 if y_pred[idx] == i else 0 for idx in true_indices]
            
            accs.append(accuracy_score(y_true_class, y_pred_class)*100)
            precisions.append(report[class_names[i]]['precision']*100)
            recalls.append(report[class_names[i]]['recall']*100)
            f1s.append(report[class_names[i]]['f1-score']*100)        
            
        # Print the metrics for each class in a table
        print(f"| {'Class':<20} | {'Accuracy':<15} | {'Precision':<15} | {'Recall':<15} | {'F1':<15} |")
        print(f"| {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} |")
        for i in range(len(class_names)):
            print(f"| {classes[i]:<20} | {accs[i]:<15.4f} | {precisions[i]:<15.4f} | {recalls[i]:<15.4f} | {f1s[i]:<15.4f} |")
    return loss_val


if __name__ == '__main__':
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # Load the model
    model = RobertaForSequenceClassification.from_pretrained("models/CodeBERT-solidifi_uncomment").to(DEVICE)    # TODO: use fine-tuned model instead
    # Load the dataset
    if os.path.exists(DATA_PATH + "/dataset.pt"):
        dataset = torch.load(DATA_PATH + "/dataset.pt", weights_only=False)
    else:
        dataset = SmardityDataset(DATA_PATH, tokenizer)
        torch.save(dataset, DATA_PATH + "/dataset.pt")
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate)
    print(len(dataloader.dataset), dataloader.dataset.labels)
    # Evaluate the model
    evaluate(model, dataloader, dataset.labels, DEVICE)
