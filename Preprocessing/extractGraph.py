import os
from utils import utils
import subprocess
from tqdm import tqdm
import argparse

funcs = utils()

def list_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files

def extractGraphFeature(file_dot):
    nodes, edges = funcs.load_dot_file(file_dot)
    graph_data = {}
    graph_data["name"] = os.path.basename(file_dot)  # Get just the filename
    graph_data["nodes"] = nodes
    graph_data["edges"] = edges

    return graph_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile EVM bytecode files and save CFG dot files.")
    parser.add_argument("--i", dest="input_dir", required=True, help="Input directory containing EVM bytecode files.")
    parser.add_argument("--o", dest="output_dir", required=True, help="Output directory to save compiled CFG dot files.")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    file_list = list_files(input_directory)
    graphs = []
    for file in tqdm(file_list):
        try:
            input_file_path = os.path.join(input_directory, file)
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(file)[0]}.dot")

            # Extract graph features
            graphs.append(extractGraphFeature(input_file_path))
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            with open("error.txt", "a") as f:
                f.write(f"Error occurred while processing {file}: {str(e)}\n")
    if output_directory[len(output_directory)-1] != "/":
        output_directory += 1
    funcs.write_dicts_to_json(graphs, output_directory+"/")
