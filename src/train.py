import os, torch
os.environ['HF_HOME'] = './cache'

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset

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
    # Setup the model parameters
    # 1. Freeze the pre-trained part of model (optional)
    for param in model.roberta.parameters():
        param.requires_grad = False
        
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.AdamW([{
        "params": model.classifier.parameters(), "lr": c_learning_rate},
        {"params": model.roberta.parameters(), "lr": r_learning_rate}])
    # Scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Training loop
    steps = 1
    for epoch in range(num_epochs):
        iterator_obj = tqdm(train_dataloader)
        for (ids, labels) in iterator_obj:
            ids = ids.to(device)
            labels = labels.to(device)
            # 1. Zero the gradients
            optimizer.zero_grad()
            with torch.autocast(device):
                # 2. Forward pass
                outputs = model(input_ids=ids, labels=labels)
            loss_val = outputs[0]
            iterator_obj.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val}")
            loss_val.backward()
            optimizer.step()
            # scheduler.step()
            # 7. Log the loss
            if steps % n_steps_to_val == 0:
                # Evaluate the model
                evaluate(model, val_dataloader, device)
                model.train()
            steps += 1
    
    # save model
    model.save_pretrained("models/CodeBERT-solidifi_finetuned")
    return model


def collate(examples):
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x[0]) for x in examples], batch_first=True, padding_value=1)
    labels = torch.tensor([x[1] for x in examples])
    return input_ids, labels


def evaluate(model, dataset:SmardityDataset, device):
    '''
    Evaluate the model on the dataset
    '''
    # TODO: Precision, Recall, F1
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    
    for ids, labels in dataset:
        ids = ids.to(device)
        labels = labels.to(device)
        y_true.extend(labels.cpu().numpy().tolist())
        with torch.no_grad(), torch.autocast(device):
            outputs = model(input_ids=ids, labels=labels)
            # 1. Compute the loss
            loss_val = outputs[0]
        y_pred.extend(outputs[1].argmax(dim=1).cpu().numpy().tolist())
        losses.append(loss_val)
    # 2. Compute the accuracy
    acc = accuracy_score(y_true, y_pred)
    # 3. Precision
    precision = precision_score(y_true, y_pred, average='macro')
    # 4. Recall
    recall = recall_score(y_true, y_pred, average='macro')
    # 5. F1
    f1 = f1_score(y_true, y_pred, average='macro')
    # 3. Compute the average loss
    loss_val = torch.tensor(losses).mean()
    # 4. Log the metrics
    print(f"Loss: {loss_val}, Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")


if __name__ == "__main__":
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "../data/train"

    torch.manual_seed(0)
    # DEVICE = 'mps'        # for mac only
    
    # Load model directly
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    if os.path.exists(DATA_PATH + "/dataset.pt"):
        dataset = torch.load(DATA_PATH + "/dataset.pt", weights_only=False)
    else:
        dataset = SmardityDataset(DATA_PATH, tokenizer)
        torch.save(dataset, DATA_PATH + "/dataset.pt")

    print(f"Dataset length: {len(dataset)}")
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=len(dataset.labels.keys())).to(DEVICE)
    print(len(dataset.labels.keys()))
    # Split the dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate)
    
    # Train the model
    model = train_model(
        model, 
        tokenizer, 
        train_dataloader, 
        val_dataloader, 
        10,
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