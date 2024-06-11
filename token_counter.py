from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

# 載入資料集
dataset_name = "HuggingFaceH4/orca_dpo_pairs"  # 請替換為你的資料集名稱
dataset = load_dataset(dataset_name, split='train_sft')

# 載入 GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 批次大小，可根據電腦記憶體調整
batch_size = 1000

# 初始化 token 計數
total_tokens = 0

# 計算批次數量
num_batches = (len(dataset) + batch_size - 1) // batch_size

# 批次化處理資料集，並使用 tqdm 顯示進度條
for i in tqdm(range(0, len(dataset), batch_size), total=num_batches, desc="Processing batches"):
    batch = dataset[i:i+batch_size]
    
    # 將 messages 中的所有訊息內容提取出來
    messages_content = [message['content'] for example in batch['messages'] for message in example]
    
    # 將訊息內容編碼為 token IDs
    token_ids = tokenizer(messages_content)['input_ids']
    
    # 計算 token 數量
    total_tokens += sum(len(ids) for ids in token_ids)

print(f"資料集 {dataset_name} 在 train_sft split 中總共有 {total_tokens} 個 tokens。")
