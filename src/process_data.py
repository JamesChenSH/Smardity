from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from typing import List, Tuple, Dict
from transformers import RobertaTokenizer

import os

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

class SmardityDataset(Dataset):
    '''
    A Solidity contract dataset. The dataset is a list of Solidity contracts, 
    each with a label of their vulnerability type, in str.
    '''

    examples: List[Tuple[str, str]]
    labels: Dict[str, int]

    def __init__(self, dataset_path, tokenizer: RobertaTokenizer):
        '''
        Construct a SmardityDataset object from the directory at dataset_path.
        The directory has the following structure:

        <dataset_path>
        ├── <vuln1>
        │   ├── file1.sol
        │   ├── file2.sol
        │   └── ...
        ├── <vuln2>
        │   ├── file1.sol
        │   ├── file2.sol
        │   └── ...
        └── ...

        Each <vuln1>, <vuln2>, are string labels of vulnerability types. We 
        will store this dict of {vuln1: 0, vuln2: 1, ...} for the labels in this 
        dataset object.

        Each file*.sol is a Solidity contract with a vulnerability type of its
        parent folder. We will read this files as text.
        
        The dataset is a list of tuples, each containing the contract and its 
        vulnerability type. 
        '''
        # TODO: Implement the constructor
        self.examples = []
        self.labels = {}
        # Get the list of directories in the dataset_path
        cls_names = []
        if '.json' in dataset_path:
            # JSON version
            import json
            with open(dataset_path, 'r') as file:
                data = json.load(file)
            for d in tqdm(data):
                contract = d['contract']
                cls_name = d['type'].upper()
                tokenized_code = tokenizer.encode(contract)
                cls_names.append(cls_name)
                cur_idx = 0
                while cur_idx < len(tokenized_code):
                    self.examples.append([tokenized_code[cur_idx:cur_idx+512], cls_name])
                    cur_idx += 512
        else:
            dirs = os.listdir(dataset_path)
            print(dataset_path, dirs)
            
            # Assign each directory a label
            for i, d in enumerate(dirs):
                if d == '.DS_Store' or '.pt' in d:
                    continue
                print(f"Parsing directory: {d}")
                cls_name = d.upper()
                cls_names.append(cls_name)
                # Get the list of files in the directory
                files = os.listdir(os.path.join(dataset_path, d))
                # Read each file as text and add it to the examples list
                for f in tqdm(files):
                    if f.endswith('.sol'):
                        with open(os.path.join(dataset_path, d, f), 'r') as file:
                            raw_code = file.read()
                        # Tokenize the code
                        tokenized_code = tokenizer.encode(raw_code) 
                        # Split the tokenized code into chunks of 512 tokens   
                        cur_idx = 0
                        while cur_idx < len(tokenized_code):
                            self.examples.append((tokenized_code[cur_idx:cur_idx+512], cls_name))
                            cur_idx += 512
                    else:
                        continue
             
        # Construct label dict   
        cls_names = list(set(cls_names))
        cls_names.sort()
        self.labels = {cls_name: i for i, cls_name in enumerate(cls_names)}
        for example in self.examples:
            example[1] = self.labels[example[1]]
        print(cls_names)

    def __len__(self):
        '''
        Return length of the dataset
        '''
        return len(self.examples)


    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.examples[i]
    