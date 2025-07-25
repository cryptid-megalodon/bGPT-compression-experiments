import os

from datasets import load_dataset

# Define the output directory
output_dir = "./uncompressed"

# Load the dataset
dataset = load_dataset("wikipedia", "20220301.en")


# Define a function to write the dataset to files
def write_dataset_to_files(dataset, split, output_dir):
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)
    for i, example in enumerate(dataset[split]):
        filename = os.path.join(output_dir, example["id"] + ".txt")
        text = example["text"]
        # Write the text to the file
        with open(filename, "w") as file:
            file.write(text)


# Write splits to files
write_dataset_to_files(dataset, "train", output_dir)
