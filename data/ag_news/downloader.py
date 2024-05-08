import os

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ag_news")


# Define a function to write the dataset to files
def write_dataset_to_files(dataset, split, output_dir):
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)
    for i, example in enumerate(dataset[split]):
        label = example["label"]
        text = example["text"]
        # Define the filename using the example index and label
        filename = os.path.join(output_dir, f"{label}_{i}.txt")
        # Write the text to the file
        with open(filename, "w") as file:
            file.write(text)


# Define the output directory
output_dir = "./uncompressed"

# Write splits to files
write_dataset_to_files(dataset, "train", output_dir)
write_dataset_to_files(dataset, "test", output_dir)
