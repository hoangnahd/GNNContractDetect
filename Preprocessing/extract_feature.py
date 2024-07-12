import re
import string
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from utils import *
import numpy as np
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process opcode data and generate graph data.")
    parser.add_argument("--i", dest="input_folder", required=True, help="Input folder containing opcode data files.")
    parser.add_argument("--o", dest="output_folder", required=True, help="Output folder to save processed graph data.")
    parser.add_argument("--label", dest="label", required=True, help="Label for contract")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    text_ds = tf.data.TextLineDataset("opcode_full.txt").filter(lambda x: tf.cast(tf.strings.length(x), bool))
    
    
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')
    
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        output_mode='int',
        output_sequence_length=2350)
    vectorize_layer.adapt(text_ds.batch(32))
    inverse_vocab = vectorize_layer.get_vocabulary()
    
    vocab2num = {}
    k = 2
    for i in range(-33,34):
        vocab2num[inverse_vocab[k]] = i
        k += 1

    jsonData = load_json_folder(input_folder)
    graphs = []
    for Graph in tqdm(jsonData):
        graph = {}
        graph["edge_index"] = Graph["edges"]
        graph["node_feature"] = []
        for node in Graph["nodes"]:
            vec = node2vec(node, vocab2num)
            vec = [str(word) for word in vec]
            graph["node_feature"].append(vec)
        graph["name"] = Graph["name"]
        graph["label"] = args.label
        graphs.append(graph)

    write_dicts_to_json(graphs, output_folder)
