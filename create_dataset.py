import os
from datasets import Dataset, Features, Value, Sequence, Image, DatasetDict
from PIL import Image as PILImage
import re
import sys

from data.config import DATASETS, DS_CONFIG


def create_vqa_dataset(images_path, queries_path, splits_path):
    """
    Create a VQA dataset from images and query files.

    Args:
        images_path: Path to directory containing images
        queries_path: Path to directory containing .txt query files
        splits_path: Splits for train, validation and test
    """

    # # Lists to store our data
    # images = []
    # questions = []
    # answers = []

    splits = {}
    splits_files = {}
    for split in splits_path.keys():
        splits[split] = []
        with open(splits_path[split], "r") as file:
            splits_files[split] = file.read().splitlines()

    # Get all query files
    query_files = [f for f in os.listdir(queries_path) if f.endswith('.txt')]

    print(f"Found {len(query_files)} query files")

    for query_file in query_files:
        # Read the query file
        query_path = os.path.join(queries_path, query_file)

        with open(query_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 2:
            print(f"Skipping {query_file}: not enough lines")
            continue

        # Extract question (all lines except last) and answer (last line)
        question_lines = [line.strip() for line in lines[:-1] if line.strip()]
        question = " ".join(question_lines)
        answer = "Yes" if lines[-1].strip() == "True" else "No"  # True or False

        # Find corresponding image
        # Assuming image has same name as query file but different extension
        query_base_name = query_file.split('.')[0]
        query_type = re.search(r"\.([a-z_]*)_[0-9]*", query_file).group(1)
        #query_type = query_base_name.split('.')[-2].split('_')[0]

        # Look for image with common extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_path = None

        for ext in image_extensions:
            potential_path = os.path.join(images_path, query_base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            print(f"Warning: No image found for {query_file}")
            continue

        # Verify image can be opened
        try:
            with PILImage.open(image_path) as img:
                img.verify()
        except Exception as e:
            print(f"Warning: Cannot open image {image_path}: {e}")
            continue

        # Add to our lists
        for split in splits_path.keys():
            if os.path.basename(image_path) + '.txt' in splits_files[split]:
                splits[split].append({
                    "image": image_path,
                    "question": question,
                    # "answers": [answer],
                    "answer": answer,
                    "query_type": query_type,
                })
                # images.append(image_path)
                # questions.append(question)
                # answers.append([answer])  # Wrap in list for Sequence

        print(f"Added: {query_base_name} to {split} (type: {query_type}) - Q: {question[:50]}... - A: {answer}")

    print(f"\nTotal samples collected:")
    for split in splits.keys():
        print(f"\t{split}: {len(splits[split])}")

    # Define the schema
    features = Features({
        "image": Image(),
        "question": Value("string"),
        "answer": Value("string"), #Sequence(Value("string")),
        "query_type": Value("string"),
    })

    # Create the dataset
    dataset = DatasetDict({
        split: Dataset.from_list(splits[split], features=features)
        for split in splits
    })

    return dataset


if __name__ == "__main__":

    # Usage
    dataset_names = ["FMT-C", "FMT-M", "Malaga", "PrimusN"]
    if len(sys.argv) > 1:
        if sys.argv[1] in dataset_names:
            dataset_name = sys.argv[1]
        else:
            raise Exception(f"Unknown dataset name: {sys.argv[1]}")
    else:
        raise AttributeError("Dataset name must be specified")

    base_dir = "." #"/home/ffuentes/Scratch/search-SMT-Noelia/"
    images_path = os.path.join(base_dir, DS_CONFIG[dataset_name]["images"])
    queries_path = os.path.join(base_dir, DS_CONFIG[dataset_name]["queries"])

    splits_path = {
        'train': os.path.join(base_dir, DS_CONFIG[dataset_name]["train"]),
        'val': os.path.join(base_dir, DS_CONFIG[dataset_name]["val"]),
        'test': os.path.join(base_dir, DS_CONFIG[dataset_name]["test"])
    }

    # Create the dataset
    dataset = create_vqa_dataset(images_path, queries_path, splits_path)

    # Check the dataset
    for split in dataset.keys():
        print(f"Dataset {split.upper()} created with {len(dataset[split])} samples")
        print(f"First sample: {dataset[split][0]}")

    # Save locally (optional)
    dataset.save_to_disk(f"./datasets/{dataset_name.lower()}-vqa")

    # Push to Hugging Face Hub (optional)
    # dataset.push_to_hub("ffuhu/fmt-c-vqa-dataset")
    dataset.push_to_hub(f"PRAIG/vqa-{dataset_name.lower()}")