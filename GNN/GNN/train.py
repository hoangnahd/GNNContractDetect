import json
import os
from tqdm import tqdm
def load_json_folder(folder_path):
    data_array = []

    # Iterate through files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a JSON file
        if filename.endswith(".json"):
            # Read and load the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Append the loaded data to the array
            data_array.append(data)

    return data_array
loaded_data = load_json_folder("train_test_data_1536")
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(loaded_data, test_size=0.2, random_state=42)

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.data import Data

class GCNBinaryClassifier(torch.nn.Module):
    def __init__(self,input_size, hidden_channels):
        super(GCNBinaryClassifier, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier for binary classification
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x).squeeze(1)  # Use a linear layer with 1 output and squeeze to get a 1D tensor

        return torch.sigmoid(x)  # Apply sigmoid activation for binary classification

# Instantiate the binary classifier model
binary_model = GCNBinaryClassifier(input_size=255, hidden_channels=32)
print(binary_model)

import torch

Train_data = []
label_1 = 0
label_0 = 0
for data in train_data:
    if data["label"] == 1:
      label_1 += 1
    else:
      label_0 += 1
    x = torch.tensor(data["node_feature"], dtype=torch.float32)
    sources = [int(num) for num in  data["edge_index"][0]]
    target = [int(num) for num in  data["edge_index"][1]]
    edges = torch.tensor([sources, target])

    y = torch.tensor([data["label"]], dtype=torch.long)

    Train_data.append(Data(x=x, edge_index=edges, y=y))
print(f"Label 1: {label_1}")
print(f"label 0: {label_0}")
Test_data = []
for data in test_data:
    x = torch.tensor(data["node_feature"], dtype=torch.float32)

    sources = [int(num) for num in  data["edge_index"][0]]
    target = [int(num) for num in  data["edge_index"][1]]
    edges = torch.tensor([sources, target])

    y = torch.tensor([data["label"]], dtype=torch.long)

    Test_data.append(Data(x=x, edge_index=edges, y=y))

from torch_geometric.loader import DataLoader

train_loader = DataLoader(Train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(Test_data, batch_size=16, shuffle=False)

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import numpy as np
import torch_geometric
from sklearn.metrics import classification_report


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        edge_index = data.edge_index
        edge_index = edge_index.to(torch.int64)
        out = model(data.x.float(), edge_index, data.batch)
        loss = criterion(out, data.y.squeeze().float())

        # Convert index tensor to int64
        index = data.edge_index.to(torch.int64)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            edge_index = torch.tensor(data.edge_index, dtype=torch.long)
            out = model(data.x.float(), edge_index, data.batch)
            predictions = (out >= 0.5).float()  # Convert probabilities to binary predictions (0 or 1)

            y_true.extend(data.y.tolist())
            y_pred.extend(predictions.tolist())

    # Convert lists to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get classification report
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore")

input_size = 1536
hidden_channels = 32
binary_model = GCNBinaryClassifier(input_size, hidden_channels)
criterion = nn.BCELoss()
optimizer = optim.Adam(binary_model.parameters(), lr=0.001)

# Initialize variables to track the best accuracy and training loss
best_accuracy = 0.0
best_train_loss = float('inf')

# Assuming you have DataLoader instances for train_loader and test_loader
for epoch in tqdm(range(100)):
    # Training
    train_loss = train(binary_model, train_loader, criterion, optimizer)

    # Evaluation
    accuracy = evaluate(binary_model, test_loader)

    # Update best accuracy and training loss
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    if train_loss < best_train_loss:
        best_train_loss = train_loss

# Print the best accuracy and training loss after the training loop
print(f"\nBest Accuracy: {best_accuracy:.4f}")
print(f"Best Training Loss: {best_train_loss:.4f}")

