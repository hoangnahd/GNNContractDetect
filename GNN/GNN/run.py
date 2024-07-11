from string import punctuation
import networkx as nx
import pydot
import numpy as np
import re
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import re
import numpy as np
import os
import nltk
import json
# Download the 'punkt' resource
nltk.download('punkt')
def extract_lines_with_pattern(input_string):
    pattern = r".*\[\w{2}\].*"  # Regular expression to match any line containing [xx]
    matches = re.findall(pattern, input_string)
    result = "\n".join(matches)
    return result
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

    # for edge in graph.edges():
    #     print(edge)
    return np.array(nodes), edges
def list_files_in_folder(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []
def node2vec(contract_data):
    contract_data = contract_data.split('\n')
    new_contract_data = [x for x in contract_data if x.strip()]
    if not new_contract_data:
        return [[0 for i in range(1536)]]  # Return a default value for an empty contract
    # Tokenize smart contract data
    tokenized_data = [word_tokenize(re.sub(r'[^\w\s]', '', contract.lower())) for contract in contract_data]

    # Initialize Word2Vec model
    model = Word2Vec(vector_size=1536, window=5, sg=1, min_count=1)  # Set vector_size to 3

    # Build vocabulary
    model.build_vocab(tokenized_data)

    # Train the Word2Vec model
    model.train(tokenized_data, total_examples=model.corpus_count, epochs=10)

    # Vectorize the sequence of opcodes in a smart contract
    vectorized_sequences = [model.wv[opcode] for opcode_sequence in tokenized_data for opcode in opcode_sequence]

    return vectorized_sequences
def write_dicts_to_json(array_of_dicts, output_directory):
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
import os

def list_folders_except_clean(directory_path):
    try:
        # Get a list of all items (files and folders) in the directory
        all_items = os.listdir(directory_path)

        # Filter out only folders and exclude the one named "clean"
        folders_except_clean = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item)) and item != "Clean"]

        return folders_except_clean

    except OSError as e:
        print(f"Error: {e}")
directory_path = 'CFG/'
vul_folder = list_folders_except_clean(directory_path)
data = []
#for name in vul_folder:
 # print(f"Loading {name} folder....")
  #dot_file_path = f'CFG/{name}/'

  #file_list = list_files_in_folder(dot_file_path)

  #for file in tqdm(file_list):
   # graph = {}
    #graph["node_feature"] = []
    #nodes, edges = load_dot_file(dot_file_path+file)
   # for node in nodes:
    #  vec = node2vec(node)
     # new_array = [sum(x) for x in zip(*vec)]
      #graph["node_feature"].append(new_array)
  #  graph["label"] = 1
   # graph["edge_index"] = np.array(edges)
   # graph["node_feature"] = np.array(graph["node_feature"])
   # graph["name"] = file
   # data.append(graph)

dot_file_path = f'CFG/Clean/'

file_list = list_files_in_folder(dot_file_path)

print("Loading Clean folder....")
for file in tqdm(file_list):
  graph = {}
  graph["node_feature"] = []
  nodes, edges = load_dot_file(dot_file_path+file)
  for node in nodes:
    vec = node2vec(node)
    new_array = [sum(x) for x in zip(*vec)]
    graph["node_feature"].append(new_array)
  graph["label"] = 0
  graph["edge_index"] = np.array(edges)
  graph["node_feature"] = np.array(graph["node_feature"])
  graph["name"] = file
  data.append(graph)
write_dicts_to_json(data,"train_test_data_1536/")
