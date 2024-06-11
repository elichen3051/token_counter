from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Variables for dataset, subset, and split
dataset_name = "Salesforce/wikitext"
subset_name = "wikitext-2-v1"
split_name = "train"

# Load the dataset from Hugging Face Hub
dataset = datasets.load_dataset(dataset_name, subset_name, split=split_name)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to count tokens in the text field
def count_tokens(examples):
    # Tokenize and count tokens, using tqdm for progress indication
    return {"num_tokens": [len(tokenizer.encode(text)) for text in tqdm(examples['text'], desc='Tokenizing')]}

# Apply function to count tokens across all examples, using multiple processors
dataset = dataset.map(count_tokens, batched=True, num_proc=4, batch_size=500)  # Adjust num_proc and batch_size as needed

# Sum up all tokens in the dataset
total_tokens = sum(dataset['num_tokens'])

# Format the total tokens with commas for readability
formatted_total_tokens = f"{total_tokens:,}"

print(f"資料集 {dataset_name} 的 {subset_name} 在 {split_name} split 中有 {formatted_total_tokens} 個 tokens.")

