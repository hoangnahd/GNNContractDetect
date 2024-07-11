import re
from tqdm import tqdm
import pydot
import numpy as np
import networkx as nx
import os
import json

class utils:
    def __init__(self):
        self.word2index = {}
        self.vocab = None
    def get_vocabulary(self, file):
        vocabulary = set()
        with open(file, 'r', encoding='utf-8') as file:
            for line in file:
                # Tokenize the line into words
                words = line.strip()
                words = words.split()
                # Add words to the vocabulary set
                vocabulary.update(words)
        self.vocab = list(vocabulary)

    def extract_lines_with_pattern(self, s):
        s = s.replace('\n', ' ')
        s = s.split("Stack size req:")
        return s[0]
    
    def load_dot_file(self, dot_file_path):
        # Read the content of the .dot file
        with open(dot_file_path, 'r') as f:
            dot_content = f.read()
        pattern = re.compile(r'digraph G {(.+?)}', re.DOTALL)
        match = pattern.search(dot_content)

        if match:
            digraph_content = match.group(1).strip()
            cleaned_text = f'digraph G {{{digraph_content}}}'
        # Load the graph from the .dot content using pydot
        graph_dot = pydot.graph_from_dot_data(cleaned_text)[0]

        # Create a directed graph using networkx
        graph = nx.DiGraph()

        # Add nodes along with their attributes to the networkx graph
        for node in graph_dot.get_nodes():
            node_name = node.get_name()
            node_attributes = node.get_attributes()
            graph.add_node(node_name, **node_attributes)
        nodes = []
        G_edges = graph.edges()

        for node in graph.nodes(data=True):
            if node[1].get("label"):
                nodes.append(self.extract_lines_with_pattern(node[1]['label']))
        count = 0
        edges = np.array([], dtype=int).reshape(0,2)
        sources = []
        target = []
        for edge in graph_dot.get_edges():
            sources.append(edge.get_source())
            target.append(edge.get_destination())
        all_edges = np.column_stack([sources, target])
        edges = np.vstack([edges, all_edges])
        edges = edges.transpose()

        return np.array(nodes), edges

    def list_files_in_folder(self, folder_path):
        try:
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            return files
        except FileNotFoundError:
            print(f"Folder not found: {folder_path}")
            return []
    
    def list_folders_except_clean(self, directory_path):
        try:
            # Get a list of all items (files and folders) in the directory
            all_items = os.listdir(directory_path)

            # Filter out only folders and exclude the one named "clean"
            folders_except_clean = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item)) and item != "Clean"]

            return folders_except_clean

        except OSError as e:
            print(f"Error: {e}")
    
    def write_dicts_to_json(self, array_of_dicts, output_directory):
        for data_dict in tqdm(array_of_dicts):
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    data_dict[key] = value.tolist()
            # Generate a unique filename for each dictionary
            name = data_dict["name"]
            filename = f"{output_directory}{name}.json"

            # Write the dictionary to the JSON file
            with open(filename, 'w') as json_file:
                json.dump(data_dict, json_file, indent=2)
    
    def load_json_folder(self, folder_path):
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

    def process_data(self, original_string):
        modified_string = re.sub(r'\d', '', original_string)
        modified_string = re.sub(r'[^\w\s]', '', modified_string)
        modified_string = re.sub(r'\s+', ' ', modified_string).strip()
        return modified_string.lower()

    def elementwise_sum(self, arr1, arr2):
        if arr1.shape != arr2.shape:
            raise ValueError("Arrays must have the same size")

        return arr1 + arr2
   
    def node2vec(self, node):
      node = self.process_data(node)
      words = node.split(' ')
      vec = np.array([0.0 for i in range(128)], dtype=np.float128)
      i = 1.0
      for word in words:
        if word in self.vocab:
            word = np.array(self.word2index[word], dtype=np.float128) / i
            vec += self.elementwise_sum(vec, word)
            i += 1
      return vec

    def inverse_vocab(self, weights):
        for i in range(len(self.vocab)):
            self.word2index[self.vocab[i]] = weights[i]
