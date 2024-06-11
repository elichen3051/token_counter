from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Variables for dataset, subset, and split
dataset_name = "allenai/c4"
subset_name = "en"
split_name = "train"

# Load the dataset from Hugging Face Hub
dataset = datasets.load_dataset(dataset_name, subset_name, split=split_name, num_proc=16)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to count tokens
def count_tokens(examples):
    # Tokenize the text and count tokens
    return {"num_tokens": [len(tokenizer.encode(text)) for text in tqdm(examples['text'], desc='Tokenizing')]}

# Apply function to count tokens across all examples, using multiple processors and larger batch size
dataset = dataset.map(count_tokens, batched=True, num_proc=16, batch_size=1000)  # Using the preferred settings

# Sum up all tokens in the dataset
total_tokens = sum(dataset['num_tokens'])

# Format the total tokens with commas for readability
formatted_total_tokens = f"{total_tokens:,}"

print(f"資料集 {dataset_name} 的 {subset_name} 在 {split_name} split 中有 {formatted_total_tokens} 個 tokens.")

