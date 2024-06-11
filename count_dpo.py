from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Variables for dataset and split
dataset_name = "HuggingFaceH4/cai-conversation-harmless"
split_name = "train_prefs"

# Load the dataset from Hugging Face Hub
dataset = datasets.load_dataset(dataset_name, split=split_name)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to concatenate fields and count tokens
def count_tokens(examples):
    token_counts = []
    for prompt, chosen_list, rejected_list in tqdm(zip(examples['prompt'], examples['chosen'], examples['rejected']),
                                                   total=len(examples['prompt']), desc='Tokenizing'):
        # Start with the prompt text
        full_text = prompt

        # Add chosen content
        for item in chosen_list:
            full_text += " " + item['content']

        # Add rejected content
        for item in rejected_list:
            full_text += " " + item['content']

        # Tokenize and count tokens
        token_counts.append(len(tokenizer.encode(full_text)))

    return {"num_tokens": token_counts}

# Apply function to count tokens across all examples, using multiple processors
dataset = dataset.map(count_tokens, batched=True, num_proc=4, batch_size=500)  # Adjust num_proc and batch_size as needed

# Sum up all tokens in the dataset
total_tokens = sum(dataset['num_tokens'])

# Format the total tokens with commas for readability
formatted_total_tokens = f"{total_tokens:,}"

print(f"資料集 {dataset_name} 在 {split_name} split 中有 {formatted_total_tokens} 個 tokens.")

