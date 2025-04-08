# utils/config.py

class Config:
    DATA_RAW_DIR = "data/raw"
    DATA_PROCESSED_DIR = "data/processed"
    MODEL_NAME = "C:\\Users\\97390\\.ollama\\models\\blobs" # 替换为 DeepSeek R1:7B 模型的实际路径
    OUTPUT_DIR = "./os_model_results"
    NUM_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    EVALUATION_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOGGING_DIR = "./logs"
    LOGGING_STEPS = 10
    FP16 = True  # 使用混合精度训练
    LOAD_IN_8BIT = True  # 使用 8-bit 量化
    TOKENIZER_NAME = "C:\\Users\\97390\\.ollama\\models\\blobs" # 通常与模型名称相同