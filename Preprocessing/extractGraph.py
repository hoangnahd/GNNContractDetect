import os
from utils import *
import subprocess
from tqdm import tqdm
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile EVM bytecode files and save CFG dot files.")
    parser.add_argument("--i", dest="input_dir", required=True, help="Input directory containing EVM bytecode files.")
    parser.add_argument("--o", dest="output_dir", required=True, help="Output directory to save compiled CFG dot files.")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    file_list = list_files_in_folder(input_directory)
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
    write_dicts_to_json(graphs, output_directory)
