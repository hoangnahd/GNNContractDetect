from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import *
from Model.GATConv import GAT
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a GAT model on graph data.")
    parser.add_argument("--data", type=str, default="data/train_test_data", help="Directory containing JSON graph data for training (default: data/train_test_data).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Choose mode: 'train' or 'test' (default: train).")

    # Arguments specific to test mode
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model file (model.pth) for testing.")
    parser.add_argument("--test_data_folder", type=str, default=None, help="Path to the folder containing test data for testing.")

    args = parser.parse_args()
    inputsize = 0
    if args.mode == "train":
        # Load data
        data = load_json_folder(args.data)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        Train_data = []
        for data in train_data:
            x = torch.tensor(data["node_feature"], dtype=torch.float64)
            inputsize = x[0]
            sources = [int(num) for num in data["edge_index"][0]]
            target = [int(num) for num in data["edge_index"][1]]
            edges = torch.tensor([sources, target])

            y = torch.tensor([data["label"]], dtype=torch.long)

            Train_data.append(Data(x=x, edge_index=edges, y=y))

        # Create DataLoader instances
        train_loader = DataLoader(Train_data, batch_size=args.batch_size, shuffle=True)

        # Initialize model, criterion, and optimizer
        model = GAT(hidden_channels=32, input_size=inputsize)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        for epoch in range(args.epochs):
            # Training
            train_loss, train_accuracy, model = train(model, optimizer, criterion, train_loader)

            # Print progress
            print(f"Epoch {epoch + 1}/{args.epochs} Train Loss: {train_loss:.4f} Train Accuracy: {train_accuracy:.4f}")

        # Save model
        torch.save(model.state_dict(), 'model/model.pth')

    elif args.mode == "test":
        # Check if model_path and test_data_folder are provided
        if args.model_path is None or args.test_data_folder is None:
            raise ValueError("In test mode, --model_path and --test_data_folder must be provided.")

        # Load test data
        Test_data = []
        test_data = load_json_folder(args.test_data_folder)
        for data in test_data:
            x = torch.tensor(data["node_feature"], dtype=torch.float64)
            sources = [int(num) for num in data["edge_index"][0]]
            target = [int(num) for num in data["edge_index"][1]]
            edges = torch.tensor([sources, target])

            y = torch.tensor([data["label"]], dtype=torch.long)

            Test_data.append(Data(x=x, edge_index=edges, y=y))

        # Create DataLoader instance for test data
        test_loader = DataLoader(Test_data, batch_size=args.batch_size, shuffle=False)

        # Initialize model and criterion
        model = GAT(hidden_channels=32, input_size=128)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        # Testing
        test_accuracy = test(model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
