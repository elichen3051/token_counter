from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Variables for dataset, subset, and split
dataset_name = "allenai/c4"
subset_name = "en"
split_name = "train"

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Stream the dataset
dataset = datasets.load_dataset(dataset_name, subset_name, split=split_name, streaming=True)

# Function to tokenize and count tokens in a stream
def count_tokens(data_stream):
    token_count = 0
    for example in tqdm(data_stream, desc="Tokenizing"):
        # Tokenize the text and count tokens
        token_count += len(tokenizer.encode(example['text']))
    return token_count

# Apply function to count tokens across all examples
total_tokens = count_tokens(dataset)

# Format the total tokens with commas for readability
formatted_total_tokens = f"{total_tokens:,}"

print(f"資料集 {dataset_name} 的 {subset_name} 在 {split_name} split 中有 {formatted_total_tokens} 個 tokens.")

