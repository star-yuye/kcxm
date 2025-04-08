# data/data_loader.py

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer
from utils.config import Config

config = Config()

def load_raw_data():
    raw_path = os.path.join(config.DATA_RAW_DIR, "os_data.jsonl")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if 'input' in json.loads(line) and 'output' in json.loads(line)]
    return data

def preprocess_data(data):
    # 分割数据集为训练集和验证集（例如 90% 训练，10% 验证）
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # 创建 Hugging Face Dataset 对象
    dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return dataset, val_dataset

def tokenize_function(examples, tokenizer, max_length=512):
    # 根据模型需求调整 max_length
    return tokenizer(
        examples['input'] + tokenizer.eos_token + examples['output'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

def prepare_datasets():
    # 加载原始数据
    raw_data = load_raw_data()
    
    # 分割数据
    train_dataset, val_dataset = preprocess_data(raw_data)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=False)
    
    # 对数据集进行分词
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # 设置格式为 PyTorch 张量
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return tokenized_train, tokenized_val, tokenizer

if __name__ == "__main__":
    tokenized_train, tokenized_val, tokenizer = prepare_datasets()
    print(f"训练集大小: {len(tokenized_train)}")
    print(f"验证集大小: {len(tokenized_val)}")
    
    # 可选：保存处理后的数据集
    tokenized_train.save_to_disk(os.path.join(config.DATA_PROCESSED_DIR, "train"))
    tokenized_val.save_to_disk(os.path.join(config.DATA_PROCESSED_DIR, "val"))
    tokenizer.save_pretrained(config.DATA_PROCESSED_DIR)