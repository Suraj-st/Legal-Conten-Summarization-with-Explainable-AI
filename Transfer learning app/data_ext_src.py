# 1. BUILD DATASET

import os
import json

# Base dataset path
dataset_path = os.path.join("dataset")
folders = ["IN-Abs", "IN-Ext", "UK-Abs"]

hf_dataset = []

def load_text_files(folder_path, full_only=False):
    """
    Loads judgement and summary text files from the given folder path.
    Returns dictionaries with filenames mapped to {text, path}.
    """
    data = {}

    # Always load judgements
    judgement_path = os.path.join(folder_path, "judgement")
    if os.path.exists(judgement_path):
        data["judgement"] = {}
        for root, _, files in os.walk(judgement_path):
            for file in sorted(files):
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dataset_path)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data["judgement"][file] = {
                            "text": f.read().strip(),
                            "path": rel_path
                        }

    # Load summaries
    summary_path = os.path.join(folder_path, "summary")
    if os.path.exists(summary_path):
        data["summary"] = {}

        # If full_only=True, target only 'summary/full'
        if full_only:
            full_path = os.path.join(summary_path, "full")
            if os.path.exists(full_path):
                for root, _, files in os.walk(full_path):
                    for file in sorted(files):
                        if file.endswith(".txt") and not file.startswith("stats-"):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, dataset_path)
                            with open(file_path, "r", encoding="utf-8") as f:
                                data["summary"][file] = {
                                    "text": f.read().strip(),
                                    "path": rel_path
                                }
        else:
            # Fallback: take summaries directly under 'summary/'
            for root, _, files in os.walk(summary_path):
                for file in sorted(files):
                    if file.endswith(".txt") and not file.startswith("stats-"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, dataset_path)
                        with open(file_path, "r", encoding="utf-8") as f:
                            data["summary"][file] = {
                                "text": f.read().strip(),
                                "path": rel_path
                            }

    return data


# Process all dataset folders
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    # IN-Abs: train-data & test-data (full_only=False — no segment-wise here)
    if folder == "IN-Abs":
        for split in ["train-data", "test-data"]:
            split_path = os.path.join(folder_path, split)
            if os.path.exists(split_path):
                split_data = load_text_files(split_path, full_only=False)
                for file in split_data.get("judgement", {}):
                    judgement_data = split_data["judgement"].get(file)
                    summary_data = split_data["summary"].get(file)
                    if judgement_data and summary_data:
                        hf_dataset.append({
                            # "source_folder": folder,
                            # "split": split,
                            # "filename": file,
                            # "judgement_path": judgement_data["path"],
                            # "summary_path": summary_data["path"],
                            "text": judgement_data["text"],
                            "summary": summary_data["text"]
                        })

    # UK-Abs: train-data & test-data, but only full summaries in test-data
    elif folder == "UK-Abs":
        for split in ["train-data", "test-data"]:
            split_path = os.path.join(folder_path, split)
            if os.path.exists(split_path):
                # Load both judgements and summaries together
                if split == "train-data":
                    split_data = load_text_files(split_path, full_only=False)
                else:  # test-data
                    split_data = load_text_files(split_path, full_only=True)  # only full summaries

                for file in split_data.get("judgement", {}):
                    judgement = split_data["judgement"].get(file, "")
                    summary = split_data["summary"].get(file, "")
                    if judgement and summary:
                        hf_dataset.append({
                            # "source_folder": folder,
                            # "split": split,
                            # "filename": file,
                            # "judgement_path": judgement["path"],
                            # "summary_path": summary["path"],
                            "text": judgement["text"],
                            "summary": summary["text"]
                        })


    # IN-Ext: only full summaries
    elif folder == "IN-Ext":
        split_data = load_text_files(folder_path, full_only=True)
        for file in split_data.get("judgement", {}):
            judgement_data = split_data["judgement"].get(file)
            summary_data = split_data["summary"].get(file)
            if judgement_data and summary_data:
                hf_dataset.append({
                    # "source_folder": folder,
                    # "split": None,
                    # "filename": file,
                    # "judgement_path": judgement_data["path"],
                    # "summary_path": summary_data["path"],
                    "text": judgement_data["text"],
                    "summary": summary_data["text"]
                })

# Final count
print(f"✅ Total usable Judgement-FullSummary pairs: {len(hf_dataset)}")

# Save to JSON
with open("legal_dataset_hf.json", "w", encoding="utf-8") as f:
    json.dump(hf_dataset, f, ensure_ascii=False, indent=2)
print("🎯 Dataset saved → legal_dataset_hf_src.json")

# Show a sample
# if hf_dataset:
#     print("\nExample entry:")
#     print(json.dumps(hf_dataset[0], indent=2, ensure_ascii=False))
