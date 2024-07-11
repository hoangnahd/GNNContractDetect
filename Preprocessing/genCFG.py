import os
import subprocess
from tqdm import tqdm
import argparse

def list_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files

def compile_and_save(directory, file_list, output_directory):
    error_files = []

    for file in tqdm(file_list):
        try:
            input_file_path = os.path.join(directory, file)
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(file)[0]}.dot")

            # Run solc command with tee
            command = f"evm-cfg {input_file_path} >> {output_file_path}"
            subprocess.run(command, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            error_files.append(file)

    if error_files:
        with open("error_Clean.txt", "a") as f:
            f.write("Errors occurred while compiling the following files:\n")
            for file in error_files:
                f.write(f"{file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile EVM bytecode files and save CFG dot files.")
    parser.add_argument("--i", dest="input_dir", required=True, help="Input directory containing EVM bytecode files.")
    parser.add_argument("--o", dest="output_dir", required=True, help="Output directory to save compiled CFG dot files.")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir

    file_list = list_files(input_directory)
    compile_and_save(input_directory, file_list, output_directory)
