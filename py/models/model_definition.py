# models/model_definition.py

from transformers import AutoModelForCausalLM
import torch
from utils.config import Config

config = Config()

def load_pretrained_model():
    try:
        # 尝试从本地路径加载模型
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            device_map="auto",
            load_in_8bit=True,  # 使用 8-bit 量化
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True  # 如果模型需要
        )
    except Exception as e:
        print(f"无法从本地加载模型: {e}")
        # 如果本地加载失败，尝试从 Hugging Face Hub 加载（确保网络连接）
        model = AutoModelForCausalLM.from_pretrained(
            "DeepSeek-R1-Distill-Qwen-7B",
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    return model