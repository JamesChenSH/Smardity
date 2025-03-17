import os, torch
os.environ['HF_HOME'] = './cache'

from transformers import RobertaTokenizer, RobertaModel
from process_data import SmardityDataset

from tqdm import tqdm

def train_model(model: RobertaModel, 
                tokenizer: RobertaTokenizer, 
                train_dataloader, 
                val_dataloader,
                num_epochs,
                learning_rate,
                warmup_steps,
                device):
    '''
    Fine tune the model on the dataset
    '''
    model.train()
    # Setup the model parameters
    # 1. Freeze the pre-trained part of model 
    for param in model.parameters():
        param.requires_grad = False
    # 2. Add classification head
    classifier = torch.nn.Linear(model.config.hidden_size, len(dataset.labels.keys()))
    model = torch.nn.Sequential(model, classifier).to(device)

    # hyperparams
    # Loss function
    loss = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        for (i, (ids, labels)) in tqdm(enumerate(train_dataloader)):
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            outputs = model(ids)
            loss_val = loss(outputs, labels)
            
            loss_val.backward()
            optimizer.step()
            scheduler.step()
            # 7. Log the loss
            if i % 10 == 0:
                print(f"Loss: {loss_val}")
        # Evaluate the model
        evaluate(model, val_dataloader)
                

def evaluate(model, dataset):
    '''
    Evaluate the model on the dataset
    '''
    loss = torch.nn.CrossEntropyLoss()
    accs = []
    losses = []
    
    for (i, (ids, labels)) in tqdm(enumerate(dataset)):
        
        with torch.no_grad():
            outputs = model(ids)
            # 1. Compute the loss
            loss_val = loss(outputs, labels)
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
    
    # Load model directly
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    
    dataset = SmardityDataset("data/contracts", tokenizer)
    print(f"Dataset length: {len(dataset)}")
    
    # Split the dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train the model
    
    
    # Create Test dataset