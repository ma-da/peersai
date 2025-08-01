import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset


# Step 1: Traverse the directory and collect .txt files
def collect_txt_files(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


# Step 2: Read the content of each .txt file
def read_txt_files(txt_files):
    texts = []
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


# Step 3: Prepare the data for fine-tuning
def prepare_dataset(texts):
    # Combine all texts into a single string
    combined_text = "\n".join(texts)

    # Tokenize the combined text
    tokenizer = GPT2Tokenizer.from_pretrained("deepseek-r1")
    tokenized_text = tokenizer(combined_text, return_tensors="pt", max_length=512, truncation=True,
                               padding="max_length")

    # Create a dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized_text['input_ids'],
        'attention_mask': tokenized_text['attention_mask']
    })

    return dataset


# Step 4: Fine-tune the deepseek-r1 model
def fine_tune_model(dataset):
    model = GPT2LMHeadModel.from_pretrained("deepseek-r1")

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("./fine-tuned-deepseek-r1")


# Main function
def main(directory):
    # Step 1: Collect .txt files
    print("Collecting txt files...")
    txt_files = collect_txt_files(directory)

    # Step 2: Read .txt files
    print("Aggregating txt files...")
    texts = read_txt_files(txt_files)
    texts_len = len(texts)
    print(f"Loaded {texts_len} bytes.")

    # Step 3: Prepare dataset
    print("Now preparing datasets...")
    dataset = prepare_dataset(texts)

    # Step 4: Fine-tune the model
    print("Fine tuning model...")
    fine_tune_model(dataset)

    print("Complete!")


if __name__ == "__main__":
    directory = "./txt_corpus/"  # Replace with your directory path
    main(directory)
