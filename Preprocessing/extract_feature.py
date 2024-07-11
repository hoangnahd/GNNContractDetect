import re
import string
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from utils import utils
import numpy as np
import argparse

def z_score_normalization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def node2vec(node):
  vec = np.array([0.0 for i in range(77)], dtype=np.float128)
  pos = 1.0
  node = funcs.process_data(node)
  words = node.split(' ')
  for word in words:
    if vocab2num.get(word):
      vec[vocab2num[word] + 33] += np.float128(vocab2num[word]) / pos
      pos += 1
  return vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process opcode data and generate graph data.")
    parser.add_argument("--i", dest="input_folder", required=True, help="Input folder containing opcode data files.")
    parser.add_argument("--o", dest="output_folder", required=True, help="Output folder to save processed graph data.")
    parser.add_argument("--label", dest="label", required=True, help="Label for contract")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    funcs = utils()

    text_ds = tf.data.TextLineDataset("opcode_full.txt").filter(lambda x: tf.cast(tf.strings.length(x), bool))
    
    
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

    sequence_length = 2350

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(text_ds.batch(32))
    inverse_vocab = vectorize_layer.get_vocabulary()
    vocab2num = {}

    k = 2
    for i in range(-33,34):
        vocab2num[inverse_vocab[k]] = i
        k += 1
    
    funcs.get_vocabulary("opcode_full.txt")

    jsonData = funcs.load_json_folder(input_folder)
    graphs = []
    for Graph in tqdm(jsonData):
        graph = {}
        graph["edge_index"] = Graph["edges"]
        graph["node_feature"] = []
        for node in Graph["nodes"]:
            vec = node2vec(node)
            vec = [str(word) for word in vec]
            graph["node_feature"].append(vec)
        graph["name"] = Graph["name"]
        graph["label"] = args.label
        graphs.append(graph)

    funcs.write_dicts_to_json(graphs, output_folder)
