import torch
from transformers import BitsAndBytesConfig
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from config import MODEL_PATH, QUANTIZATION_CONFIG


def load_model_and_processor():
    """加载模型和处理器"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=QUANTIZATION_CONFIG,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

    return model, processor


def process_vision_info(messages):
    """处理视觉信息（具体实现）"""
    # 这里是实际实现，与原始代码相同