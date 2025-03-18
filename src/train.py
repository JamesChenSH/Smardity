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
    
    # save model
    model.save_pretrained("models/CodeBERT-solidifi")
    return model
                

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
    model = train_model(model, tokenizer, train_dataloader, val_dataloader, 1, 1e-6, 100, "cuda")

# %%
    # Test one
    prompt_buggy_code = """/**
 * Source Code first verified at https://etherscan.io on Monday, April 15, 2019
 (UTC) */

pragma solidity ^0.5.8;

contract ERC20Interface {
    function balanceOf(address from) public view returns (uint256);
mapping(address => uint) redeemableEther_re_ent39;
function claimReward_re_ent39() public {
        // ensure there is a reward to give
        require(redeemableEther_re_ent39[msg.sender] > 0);
        uint transferValue_re_ent39 = redeemableEther_re_ent39[msg.sender];
        msg.sender.transfer(transferValue_re_ent39);   //bug
        redeemableEther_re_ent39[msg.sender] = 0;
    }
    function transferFrom(address from, address to, uint tokens) public returns (bool);
mapping(address => uint) balances_re_ent38;
function withdrawFunds_re_ent38 (uint256 _weiToWithdraw) public {
        require(balances_re_ent38[msg.sender] >= _weiToWithdraw);
        // limit the withdrawal
        require(msg.sender.send(_weiToWithdraw));  //bug
        balances_re_ent38[msg.sender] -= _weiToWithdraw;
    }
    function allowance(address owner, address spender) public view returns (uint256);
mapping(address => uint) balances_re_ent10;
function withdrawFunds_re_ent10 (uint256 _weiToWithdraw) public {
        require(balances_re_ent10[msg.sender] >= _weiToWithdraw);
        // limit the withdrawal
        require(msg.sender.send(_weiToWithdraw));  //bug
        balances_re_ent10[msg.sender] -= _weiToWithdraw;
    }
    function burn(uint256 amount) public;
mapping(address => uint) userBalance_re_ent12;
function withdrawBalance_re_ent12() public{
        // send userBalance[msg.sender] ethers to msg.sender
        // if mgs.sender is a contract, it will call its fallback function
        if( ! (msg.sender.send(userBalance_re_ent12[msg.sender]) ) ){
            revert();
        }
        userBalance_re_ent12[msg.sender] = 0;
    }
}


contract UsernameRegistry {
mapping(address => uint) balances_re_ent29;
    function withdraw_balances_re_ent29 () public {
       if (msg.sender.send(balances_re_ent29[msg.sender ]))
          balances_re_ent29[msg.sender] = 0;
      }

event Register(address indexed _owner, bytes32 _name, bytes32 _userId);
mapping(address => uint) redeemableEther_re_ent4;
function claimReward_re_ent4() public {
        // ensure there is a reward to give
        require(redeemableEther_re_ent4[msg.sender] > 0);
        uint transferValue_re_ent4 = redeemableEther_re_ent4[msg.sender];
        msg.sender.transfer(transferValue_re_ent4);   //bug
        redeemableEther_re_ent4[msg.sender] = 0;
    }

ERC20Interface public manaToken;uint256 counter_re_ent42 =0;
function callme_re_ent42() public{
        require(counter_re_ent42<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent42 += 1;
    }

uint256 public price = 100000000000000000000;mapping(address => uint) userBalance_re_ent5;
function withdrawBalance_re_ent5() public{
        // send userBalance[msg.sender] ethers to msg.sender
        // if mgs.sender is a contract, it will call its fallback function
        if( ! (msg.sender.send(userBalance_re_ent5[msg.sender]) ) ){
            revert();
        }
        userBalance_re_ent5[msg.sender] = 0;
    }

mapping (bytes32 => address) nameToAddress;mapping(address => uint) balances_re_ent8;
    function withdraw_balances_re_ent8 () public {
       (bool success,) = msg.sender.call.value(balances_re_ent8[msg.sender ])("");
       if (success)
          balances_re_ent8[msg.sender] = 0;
      }

mapping (bytes32 => address) userIdToAddress;address payable lastPlayer_re_ent9;
      uint jackpot_re_ent9;
	  function buyTicket_re_ent9() public{
	    (bool success,) = lastPlayer_re_ent9.call.value(jackpot_re_ent9)("");
	    if (!success)
	        revert();
      lastPlayer_re_ent9 = msg.sender;
      jackpot_re_ent9    = address(this).balance;
    }

mapping (address => bytes32) public name;mapping(address => uint) redeemableEther_re_ent11;
function claimReward_re_ent11() public {
        // ensure there is a reward to give
        require(redeemableEther_re_ent11[msg.sender] > 0);
        uint transferValue_re_ent11 = redeemableEther_re_ent11[msg.sender];
        msg.sender.transfer(transferValue_re_ent11);   //bug
        redeemableEther_re_ent11[msg.sender] = 0;
    }

address public owner;

constructor(ERC20Interface _mana) public {
    manaToken = _mana;
    owner = msg.sender;
}
bool not_called_re_ent13 = true;
function bug_re_ent13() public{
        require(not_called_re_ent13);
        (bool success,)=msg.sender.call.value(1 ether)("");
        if( ! success ){
            revert();
        }
        not_called_re_ent13 = false;
    }

modifier onlyOwner {
    require(msg.sender == owner);
    _;
}

function registerUsername(address _targetAddress, bytes32 _name, bytes32 _userId) onlyOwner external {
    _requireBalance();
    require(isNameAvailable(_name), "The name was already taken");
    require(isUserIdAvailable(_userId), "The userId already has a name");

    manaToken.transferFrom(_targetAddress, address(this), price);
    manaToken.burn(price);

    nameToAddress[_name] = _targetAddress;
    userIdToAddress[_userId] = _targetAddress;
    name[_targetAddress] = _name;

    emit Register(_targetAddress, _userId, _name);
}
mapping(address => uint) balances_re_ent17;
function withdrawFunds_re_ent17 (uint256 _weiToWithdraw) public {
        require(balances_re_ent17[msg.sender] >= _weiToWithdraw);
        // limit the withdrawal
        (bool success,)=msg.sender.call.value(_weiToWithdraw)("");
        require(success);  //bug
        balances_re_ent17[msg.sender] -= _weiToWithdraw;
    }

function isNameAvailable(bytes32 _name) public view returns (bool) {
    return nameToAddress[_name] == address(0);
}
address payable lastPlayer_re_ent16;
      uint jackpot_re_ent16;
	  function buyTicket_re_ent16() public{
	    if (!(lastPlayer_re_ent16.send(jackpot_re_ent16)))
        revert();
      lastPlayer_re_ent16 = msg.sender;
      jackpot_re_ent16    = address(this).balance;
    }
function isUserIdAvailable(bytes32 _name) public view returns (bool) {
    return userIdToAddress[_name] == address(0);
}
uint256 counter_re_ent28 =0;
function callme_re_ent28() public{
        require(counter_re_ent28<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent28 += 1;
    }

// Lack of security (set only owner)
function setPrice(uint256 _price) onlyOwner public {
    price = _price;
}
uint256 counter_re_ent14 =0;
function callme_re_ent14() public{
        require(counter_re_ent14<=5);
	if( ! (msg.sender.send(10 ether) ) ){
            revert();
        }
        counter_re_ent14 += 1;
    }

function _requireBalance() internal view {
    require(
        manaToken.balanceOf(msg.sender) >= price,
        "Insufficient funds"
    );
    require(
        manaToken.allowance(msg.sender, address(this)) >= price,
        "The contract is not authorized to use MANA on sender behalf"
    );
}
mapping(address => uint) balances_re_ent15;
    function withdraw_balances_re_ent15 () public {
       if (msg.sender.send(balances_re_ent15[msg.sender ]))
          balances_re_ent15[msg.sender] = 0;
      }
}"""
    test_input_ids = tokenizer.encode(prompt_buggy_code)
    i = 0
    prompts = []
    while i < len(test_input_ids):
        prompts.append(test_input_ids[i:i+512])
        i += 512
    print(model(prompt_buggy_code))