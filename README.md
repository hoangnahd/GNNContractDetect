## Introduction
This project aims to leverage Graph Neural Networks (GNNs) to detect vulnerabilities in smart contracts. By analyzing the structure and code of smart contracts, we can identify potential security issues and improve the overall safety of blockchain applications. We use the [evm-cfg](https://github.com/plotchy/evm-cfg/) tool to generate Control Flow Graphs (CFG) from Ethereum smart contracts.
## Features
**Smart Contract Parsing:** Convert smart contract code into graph representations using the [evm-cfg](https://github.com/plotchy/evm-cfg/) tool.\
Graph Neural Networks: Utilize GNN models to analyze the structure and relationships within the smart contract.\
**Vulnerability Detection:** Identify common vulnerabilities in smart contracts such as reentrancy, arithmetic.\
## Folder Structure
### Preprocessing/: Contains scripts for:
**genCFG:** Obtain CFGs using [evm-cfg](https://github.com/plotchy/evm-cfg/).\
**extractGraph:** Generate graph representations from smart contracts.\
**extract_feature:** Extract features for GNN training and evaluation.
### data/: 
Contains datasets of smart contracts for training, graph data and their CFGs.
## Usage
Run the script in different modes: train, test. Below are the examples of how to use each mode.

### 1. Training Mode
python run.py --mode train --data data/train_test_data --batch_size 32 --learning_rate 0.001 --epochs 30 

### 2. Test Mode
python run.py --mode test --model_path model/model.pth --test_data_folder data/test_data

### 3. Arguments
**--data:** Directory containing JSON graph data for training (default: data/train_test_data).\
**--batch_size:** Batch size for DataLoader (default: 32).\
**--learning_rate:** Learning rate for optimizer (default: 0.001).\
**--epochs:** Number of epochs for training (default: 30).\
**--mode:** Choose mode: 'train' or 'test' (default: 'train').\
**--model_path:** Path to the trained model file (model.pth) for testing (required for test mode).\
**--test_data_folder**: Path to the folder containing test data for testing (required for test mode).
