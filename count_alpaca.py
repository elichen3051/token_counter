from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

# 載入資料集
dataset_name = "tatsu-lab/alpaca"
dataset = load_dataset(dataset_name, split='train')

# 載入 GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 批次大小,可根據電腦記憶體調整
batch_size = 1000

# 初始化 token 計數
total_tokens = 0

# 計算批次數量
num_batches = (len(dataset) + batch_size - 1) // batch_size

# 批次化處理資料集,並使用 tqdm 顯示進度條
for i in tqdm(range(0, len(dataset), batch_size), total=num_batches, desc="Processing batches"):
    batch = dataset[i:i+batch_size]
    
    # 將 "text" 欄位中的文本編碼為 token IDs
    token_ids = tokenizer(batch['text'])['input_ids']
    
    # 計算 token 數量
    total_tokens += sum(len(ids) for ids in token_ids)

print(f"資料集 {dataset_name} 在 train split 中總共有 {total_tokens} 個 tokens。")
