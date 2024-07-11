from tqdm import tqdm
import os
import json
import torch

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
def convert_to_one_hot(label_tensor, num_classes):
    """
    Convert a label tensor to its one-hot encoded representation.

    Args:
    - label_tensor (torch.Tensor): Tensor containing the labels.
    - num_classes (int): Number of classes.

    Returns:
    - torch.Tensor: One-hot encoded tensor.
    """
    # Initialize the one-hot encoded tensor
    one_hot_tensor = torch.zeros(len(label_tensor), num_classes)

    # Fill in the one-hot encoded tensor
    one_hot_tensor[range(len(label_tensor)), label_tensor] = 1

    return one_hot_tensor
def train(model, optimizer, criterion, loader):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data in loader:  # Iterate in batches over the train dataset.
        data.x = data.x.to(model.parameters().__next__().dtype)
        data.edge_index = data.edge_index.to(torch.long)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, convert_to_one_hot(data.y,3))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        # Compute loss and accuracy
        total_loss += loss.item()
        predictions = out.argmax(dim=1)
        correct_predictions += (predictions == data.y).sum().item()
        total_samples += data.y.size(0)

    # Calculate average loss and accuracy
    average_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy, model
def test(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.edge_index = data.edge_index.to(torch.long)
        data.x = data.x.to(model.parameters().__next__().dtype)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        y_true.extend(data.y.tolist())
        y_pred.extend(pred.tolist())
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    # Compute accuracy
    accuracy = correct / len(loader.dataset)

    return accuracy