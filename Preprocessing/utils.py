import re
from tqdm import tqdm
import pydot
import numpy as np
import networkx as nx
import os
import json


def extract_lines_with_pattern(self, s):
    s = s.replace('\n', ' ')
    s = s.split("Stack size req:")
    return s[0]

def load_dot_file(dot_file_path):
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
            nodes.append(extract_lines_with_pattern(node[1]['label']))
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

def list_files_in_folder(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []

def write_dicts_to_json(array_of_dicts, output_directory):
    for data_dict in tqdm(array_of_dicts):
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                data_dict[key] = value.tolist()
        # Generate a unique filename for each dictionary
        name = data_dict["name"]
        filename = f"{os.path.join(output_directory, name)}.json"

        # Write the dictionary to the JSON file
        with open(filename, 'w') as json_file:
            json.dump(data_dict, json_file, indent=2)

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

def process_data(original_string):
    modified_string = re.sub(r'\d', '', original_string)
    modified_string = re.sub(r'[^\w\s]', '', modified_string)
    modified_string = re.sub(r'\s+', ' ', modified_string).strip()
    return modified_string.lower()

def elementwise_sum(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same size")

    return arr1 + arr2

def node2vec(node, vocab2num):
  vec = np.array([0.0 for i in range(77)], dtype=np.float128)
  pos = 1.0
  node = process_data(node)
  words = node.split(' ')
  for word in words:
    if vocab2num.get(word):
      vec[vocab2num[word] + 33] += np.float128(vocab2num[word]) / pos
      pos += 1
  return vec

def z_score_normalization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def extractGraphFeature(file_dot):
    nodes, edges = load_dot_file(file_dot)
    graph_data = {}
    graph_data["name"] = os.path.basename(file_dot)  # Get just the filename
    graph_data["nodes"] = nodes
    graph_data["edges"] = edges

    return graph_data