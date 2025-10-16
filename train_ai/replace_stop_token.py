"""
This script changes qa pairs files with json content with stop tokens at the end of answers
to append a newline before the stop token </s>. This should help with run-on sentences.
Note do not run on a qa pairs corpus that already has newlines!
"""

import json
import os
import glob

# Directory containing the JSON files (modify this path as needed)
directory = "/Volumes/SSK/peers_dev/corpus/wtk_beta_slim_qa_pairs2"  # Current directory, or specify your directory path

# Find all JSON files in the directory
json_files = glob.glob(os.path.join(directory, "*.json"))

for json_file in json_files:
    print(f"Replacing stop tokens in file {json_file}...")
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process each question-answer pair
        for item in data:
            if "answer" in item and item["answer"].endswith("</s>"):
                # Replace </s> with \n</s>
                item["answer"] = item["answer"][:-4] + "\n</s>"

        # Write the modified data back to the same file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Processed file: {json_file}")

    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")

print("All files processed.")