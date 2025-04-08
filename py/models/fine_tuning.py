# models/fine_tuning.py

import os
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import torch
from models.model_definition import load_pretrained_model
from utils.config import Config

config = Config()

def fine_tune_model():
    # 加载预训练模型和分词器
    model, tokenizer = load_pretrained_model()
    
    # 加载处理后的数据集
    processed_dir = config.DATA_PROCESSED_DIR
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed data directory not found at {processed_dir}. 请先运行数据准备脚本。")
    
    tokenized_train = load_from_disk(os.path.join(processed_dir, "train"))
    tokenized_val = load_from_disk(os.path.join(processed_dir, "val"))
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy=config.EVALUATION_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
        fp16=config.FP16,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none"  # 避免与某些日志系统冲突
    )
    
    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,  # 传递 tokenizer
        # compute_metrics=None  # 可以根据需要定义评估指标
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    final_output_dir = os.path.join(config.OUTPUT_DIR, "final")
    os.makedirs(final_output_dir, exist_ok=True)  # 确保目录存在
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"模型已保存到 {final_output_dir}")

if __name__ == "__main__":
    fine_tune_model()