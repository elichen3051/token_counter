from transformers import GPT2Tokenizer
import datasets
from tqdm.auto import tqdm

# Load the dataset from Hugging Face Hub
dataset = datasets.load_dataset("yahma/alpaca-cleaned", split='train')

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to concatenate fields and count tokens
def count_tokens(examples):
    # Concatenate the instruction, input (if it exists), and output
    texts = []
    for instr, inp, outp in zip(examples['instruction'], examples['input'], examples['output']):
        full_text = instr
        if inp is not None:  # Check if 'input' is not empty
            full_text += " " + inp
        full_text += " " + outp
        texts.append(full_text)
    
    # Tokenize and count tokens, using tqdm for progress indication
    token_counts = [len(tokenizer.encode(text)) for text in tqdm(texts, desc='Tokenizing')]
    return {"num_tokens": token_counts}

# Apply function to count tokens across all examples, using multiple processors
dataset = dataset.map(count_tokens, batched=True, num_proc=4, batch_size=500)  # Adjust num_proc and batch_size as needed

# Sum up all tokens in the dataset
total_tokens = sum(dataset['num_tokens'])

print(f"Total training tokens in the dataset: {total_tokens}")

