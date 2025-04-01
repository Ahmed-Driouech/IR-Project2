import os
import datasets
from tqdm import tqdm

def download_msmarco_dataset():
    os.makedirs("msmarco_data", exist_ok=True)
    msmarco = datasets.load_dataset("ms_marco", "v2.1", split="validation")

    output_path = os.path.join("msmarco_data", "validation-00000-of-00001.parquet")
    msmarco.to_parquet(output_path)

    return output_path

def main():
    validation_path = download_msmarco_dataset()

if __name__ == "__main__":
    main()
