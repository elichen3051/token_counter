from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Variables for dataset and split
dataset_name = "bigbio/med_qa"
split_name = "train"

# Load the dataset from Hugging Face Hub
dataset = datasets.load_dataset(dataset_name, split=split_name)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to count tokens
def count_tokens(examples):
    # Combine 'question', 'answer', and 'options' into one text for each example
    token_counts = []
    for question, answer, options in tqdm(zip(examples['question'], examples['answer'], examples['options']), 
                                          total=len(examples['question']), desc='Tokenizing'):
        # Start with the question text
        full_text = question

        # Add options content
        options_text = ' '.join([option['key'] + ': ' + option['value'] for option in options])
        full_text += " " + options_text

        # Add the answer
        full_text += " " + answer

        # Tokenize and count tokens
        token_counts.append(len(tokenizer.encode(full_text)))

    return {"num_tokens": token_counts}

# Apply function to count tokens across all examples, using multiple processors and larger batch size
dataset = dataset.map(count_tokens, batched=True, num_proc=16, batch_size=1000)  # Using the preferred settings

# Sum up all tokens in the dataset
total_tokens = sum(dataset['num_tokens'])

# Format the total tokens with commas for readability
formatted_total_tokens = f"{total_tokens:,}"

print(f"資料集 {dataset_name} 在 {split_name} split 中有 {formatted_total_tokens} 個 tokens.")

