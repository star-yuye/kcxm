# train.py

import os
from data.data_loader import prepare_datasets
from models.fine_tuning import fine_tune_model
from utils.config import Config

config = Config()

def main():
    # 检查处理后的数据目录是否存在
    processed_dir = config.DATA_PROCESSED_DIR
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"创建处理后的数据目录: {processed_dir}")
    else:
        print(f"使用现有的处理后数据目录: {processed_dir}")
    
    # 数据准备
    print("开始数据准备...")
    try:
        tokenized_train, tokenized_val, tokenizer = prepare_datasets()
        print("数据准备完成。")
    except Exception as e:
        print(f"数据准备失败: {e}")
        return
    
    # 模型微调
    print("开始模型微调...")
    try:
        fine_tune_model(tokenized_train, tokenized_val, tokenizer)
        print("模型微调完成。")
    except Exception as e:
        print(f"模型微调失败: {e}")

if __name__ == "__main__":
    main()