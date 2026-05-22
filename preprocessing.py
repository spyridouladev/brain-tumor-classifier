import os
import splitfolders
import shutil
from libs.utils import get_base_path

RATIOS = (0.7,0.2,0.1)
SEED = 1337

def prepare_output_directory(output_path):
    # Create or write into output directory
    if os.path.exists(output_path):
        print("Output directory already exists.")
        # Delete all contents of the directory.
        print("Clearing output directory.")
        shutil.rmtree(output_path)

    # Create the directory again without any contents.
    os.makedirs(output_path)
    print("Output directory created successfully.")

def split_dataset(input_path,output_path,ratios,seed):
    splitfolders.ratio(
        input_path,
        output=output_path,
        seed=seed,
        ratio=ratios
    )
    print("Dataset has been split into training, validation and testing.")

def preprocessing(ratios=RATIOS,seed=SEED):
    # Set base path as the parent directory of the .py file.
    base_path = get_base_path()

    input_path = os.path.abspath(os.path.join(base_path, "tumor-set"))
    output_path = os.path.join(base_path, "dataset_split")

    prepare_output_directory(output_path)

    split_dataset(input_path,output_path,ratios,seed)

if __name__ == "__main__":
    preprocessing()