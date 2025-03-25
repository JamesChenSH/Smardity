import os, torch
os.environ['HF_HOME'] = './cache'

import torch.utils.data.dataloader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from tqdm import tqdm

def train_model(model: RobertaForSequenceClassification, 
                tokenizer: RobertaTokenizer, 
                train_dataloader, 
                val_dataloader,
                num_epochs,
                c_learning_rate,
                r_learning_rate,
                n_steps_to_val,
                device):
    '''
    Fine tune the model on the dataset
    '''
    model.train()
    # Freeze the pre-trained part of the model (optional)
    for param in model.roberta.parameters():
        param.requires_grad = False
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": c_learning_rate},
        {"params": model.roberta.parameters(), "lr": r_learning_rate}
    ])

    # Training loop
    steps = 1
    for epoch in range(num_epochs):
        iterator_obj = tqdm(train_dataloader)
        for ids, attention_mask, labels in iterator_obj:
            ids = ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device):
                outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labels)
            loss_val = outputs[0]
            iterator_obj.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val.item():.4f}")
            loss_val.backward()
            optimizer.step()
            if steps % n_steps_to_val == 0:
                evaluate(model, val_dataloader, device)
                model.train()
            steps += 1
    
    model.save_pretrained("models/CodeBERT-solidifi_finetuned")
    return model

def collate(examples):
    '''
    Collate function to prepare batches with attention masks
    '''
    pad_token_id = 1
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x[0]) for x in examples], 
        batch_first=True, 
        padding_value=pad_token_id
    )
    labels = torch.tensor([x[1] for x in examples])
    attention_mask = (input_ids != pad_token_id).long()
    return input_ids, attention_mask, labels

def evaluate(model, dataloader: torch.utils.data.dataloader.DataLoader, device):
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
    
    # Get class names
    class_names = list(dataloader.dataset.labels.keys())
    val_to_class = {v: k for k, v in dataloader.dataset.labels.items()}
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    
    
    
    
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
        
    for i in range(len(class_names)):
        y_pred_class = [1 if y == i else 0 for y in y_pred]
        y_true_class = [1 if y == i else 0 for y in y_true]
        
        precisions.append(precision_score(y_true_class, y_pred_class, zero_division=0)*100)
        recalls.append(recall_score(y_true_class, y_pred_class, zero_division=0)*100)
        f1s.append(f1_score(y_true_class, y_pred_class, zero_division=0)*100)
        
        
    
    # 3. Compute the average loss
    loss_val = torch.tensor(losses).mean()
    # 4. Log the metrics
    print(f"Loss: {loss_val}, Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    # Print the metrics for each class in a table
    print(f"| {'Class':<20} | {'Accuracy':<15} | {'Precision':<15} | {'Recall':<15} | {'F1':<15} |")
    print(f"| {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} |")
    for i in range(len(class_names)):
        print(f"| {classes[i]:<20} | {accs[i]:<15.4f} | {precisions[i]:<15.4f} | {recalls[i]:<15.4f} | {f1s[i]:<15.4f} |")
    return loss_val, acc, precision, recall, f1

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        
    DATA_PATH = "../data/train"
    torch.manual_seed(0)
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Load or create dataset
    if os.path.exists(DATA_PATH + "/dataset.pt"):
        dataset = torch.load(DATA_PATH + "/dataset.pt", weights_only=False)
    else:
        dataset = SmardityDataset(DATA_PATH, tokenizer)
        torch.save(dataset, DATA_PATH + "/dataset.pt")

    print(f"Dataset length: {len(dataset)}")
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", 
        num_labels=len(dataset.labels.keys())
    ).to(DEVICE)
    print(len(dataset.labels.keys()))
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with updated collate
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=lambda x: collate(x, tokenizer)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=lambda x: collate(x, tokenizer)
    )
    
    # Train the model
    model = train_model(
        model, 
        tokenizer, 
        train_dataloader, 
        val_dataloader, 
        num_epochs=3,
        c_learning_rate=1e-3, 
        r_learning_rate=5e-6,
        n_steps_to_val=500,
        device=DEVICE
    )

# %%
    # Test one

    # model = RobertaForSequenceClassification.from_pretrained("models/CodeBERT-solidifi").to(DEVICE)
    # with open('./sample.sol', 'r') as file:
    #     prompt_buggy_code = file.read()
    # test_input_ids = tokenizer.encode(prompt_buggy_code)
    # i = 0
    # prompts = []
    # while i < len(test_input_ids):
    #     prompts.append(test_input_ids[i:i+512])
    #     i += 512
    # prompts = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in prompts], batch_first=True, padding_value=1)
    # print(model(prompts.to(DEVICE)).logits.argmax(dim=1))