import os
from datasets import load_dataset

# Load the dataset from local path
dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "multiconer_v2")
dataset = load_dataset(dataset_path, "Arabic (AR)")
