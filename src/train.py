import os, torch
os.environ['HF_HOME'] = './cache'

import torch.utils.data.dataloader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from process_data import SmardityDataset, collate
from evaluate import evaluate, collate
from tqdm import tqdm


def train_model(model: RobertaForSequenceClassification, 
                tokenizer: RobertaTokenizer, 
                train_dataloader, 
                val_dataloader,
                num_epochs,
                c_learning_rate,
                r_learning_rate,
                n_steps_to_val,
                output_path="models/CodeBERT-solidifi",
                device="cuda"):
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
                evaluate(model, val_dataloader, device=device)
                model.train()
            steps += 1
    
    model.save_pretrained(output_path)
    return model


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
    # if os.path.exists(DATA_PATH + "/dataset.pt"):
    #     dataset = torch.load(DATA_PATH + "/dataset.pt", weights_only=False)
    # else:
    #     dataset = SmardityDataset(DATA_PATH, tokenizer)
    #     torch.save(dataset, DATA_PATH + "/dataset.pt")

    DATA_JSON = "../data/train/clean_labeled_contracts.json"
    if os.path.exists(DATA_PATH + "/dataset_uncomment.pt"):
        dataset = torch.load(DATA_PATH + "/dataset_uncomment.pt", weights_only=False)
    else:
        dataset = SmardityDataset(DATA_JSON, tokenizer)
        torch.save(dataset, DATA_PATH + "/dataset_uncomment.pt")

    print(f"Dataset length: {len(dataset)}")
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", 
        num_labels=len(dataset.labels.keys())
    ).to(DEVICE)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with updated collate
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate
    )
    
    # Train the model
    model = train_model(
        model, 
        tokenizer, 
        train_dataloader, 
        val_dataloader, 
        num_epochs=10,
        c_learning_rate=1e-3, 
        r_learning_rate=5e-6,
        n_steps_to_val=100,
        output_path="models/CodeBERT-solidifi_uncomment",
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