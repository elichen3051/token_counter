from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Variables for dataset, subset, and split
dataset_name = "abisee/cnn_dailymail"
subset_name = "3.0.0"
split_name = "train"

# Load the dataset from Hugging Face Hub
dataset = datasets.load_dataset(dataset_name, subset_name, split=split_name)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to count tokens for both 'article' and 'highlights'
def count_tokens(examples):
    # Combine 'article' and 'highlights' and tokenize
    texts = [article + " " + highlight for article, highlight in zip(examples['article'], examples['highlights'])]
    # Tokenize and count tokens, using tqdm for progress indication
    return {"num_tokens": [len(tokenizer.encode(text)) for text in tqdm(texts, desc='Tokenizing')]}

# Apply function to count tokens across all examples, using multiple processors
dataset = dataset.map(count_tokens, batched=True, num_proc=4, batch_size=500)  # Adjust num_proc and batch_size as needed

# Sum up all tokens in the dataset
total_tokens = sum(dataset['num_tokens'])

# Format the total tokens with commas for readability
formatted_total_tokens = f"{total_tokens:,}"

print(f"資料集 {dataset_name} 的 {subset_name} 在 {split_name} split 中有 {formatted_total_tokens} 個 tokens.")

