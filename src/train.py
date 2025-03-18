import os, torch
os.environ['HF_HOME'] = './cache'

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset

from tqdm import tqdm

def train_model(model: RobertaForSequenceClassification, 
                tokenizer: RobertaTokenizer, 
                train_dataloader, 
                val_dataloader,
                num_epochs,
                learning_rate,
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
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Training loop
    steps = 1
    for epoch in range(num_epochs):
        for (ids, labels) in tqdm(train_dataloader):
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            ids = ids.to(device)
            labels = labels.to(device)
            # 2. Forward pass
            outputs = model(input_ids=ids, labels=labels)
            loss_val = outputs[0]
            
            loss_val.backward()
            optimizer.step()
            scheduler.step()
            # 7. Log the loss
            if steps % n_steps_to_val == 0:
                # Evaluate the model
                evaluate(model, val_dataloader)
                model.train()
                print(f"Loss: {loss_val}")
            steps += 1
    
    # save model
    model.save_pretrained("models/CodeBERT-solidifi")
    return model


def collate(examples):
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x[0]) for x in examples], batch_first=True, padding_value=1)
    labels = torch.tensor([x[1] for x in examples])
    return input_ids, labels


def evaluate(model, dataset, device):
    '''
    Evaluate the model on the dataset
    '''
    model.eval()
    accs = []
    losses = []
    
    for (i, (ids, labels)) in tqdm(enumerate(dataset)):
        ids = ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids=ids, labels=labels)
            # 1. Compute the loss
            loss_val = outputs[0]
            # 2. Compute the accuracy
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            accs.append(accuracy)
            losses.append(loss_val)
        
    # 3. Compute the average loss and accuracy
    loss_val = torch.tensor(losses).mean()
    acc = torch.tensor(accs).mean()
    # 4. Log the loss and accuracy
    print(f"Loss: {loss_val}, Accuracy: {acc}")


if __name__ == "__main__":
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = 'mps'
    
    # Load model directly
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    dataset = SmardityDataset("../data", tokenizer)
    print(f"Dataset length: {len(dataset)}")
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=len(dataset.labels.keys())).to(DEVICE)
    
    # Split the dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)
    
    # Train the model
    model = train_model(model, tokenizer, train_dataloader, val_dataloader, 1, 1e-6, 100, DEVICE)

# %%
    # Test one
    with open('./sample.sol', 'r') as file:
        prompt_buggy_code = file.read()
    test_input_ids = tokenizer.encode(prompt_buggy_code)
    i = 0
    prompts = []
    while i < len(test_input_ids):
        prompts.append(test_input_ids[i:i+512])
        i += 512
    print(model(prompt_buggy_code))