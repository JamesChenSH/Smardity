from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class SmardityDataset(Dataset):
    '''
    A Solidity contract dataset. The dataset is a list of Solidity contracts, 
    each with a label of their vulnerability type, in str.
    '''

    examples: List[Tuple[str, str]]
    labels: Dict[str, int]

    def __init__(self, dataset_path):
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
        

    def __len__(self):
        '''
        Return length of the dataset
        '''
        return len(self.examples)


    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.examples[i]
    
