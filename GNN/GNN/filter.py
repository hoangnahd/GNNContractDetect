import os
import json

def process_files(folder_path, remove_condition):
    files_to_remove = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    
                    # Check if 'label' key exists and its value meets the condition
                    if 'label' in data and data['label'] == remove_condition:
                        files_to_remove.append(file_path)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_path}")

    # Remove the specified number of files
    for file_to_remove in files_to_remove[:300]:
        os.remove(file_to_remove)

# Replace 'your_folder_path' with the path to your folder containing JSON files
folder_path = 'train_test_data_1536'

# Remove files where 'label' key exists and its value is 1 (700 files)
process_files(folder_path, 0)


