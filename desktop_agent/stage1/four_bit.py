from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import BitsAndBytesConfig
import time

# 指定本地模型文件夹路径
model_path = "D:\Downloads\Model\Qwen2.5-VL-3B-Instruct"

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4-bit量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
    bnb_4bit_quant_type="nf4",  # 使用NormalFloat4量化
    bnb_4bit_use_double_quant=True  # 嵌套量化进一步压缩
)

# 加载模型时指定local_files_only=True，并应用量化配置
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True  # 只从本地加载模型文件
)

# 加载分词器时指定local_files_only=True
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,  # 只从本地加载分词器文件
    #use_fast=True  # 强制启用快速处理器
)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "D:\project-file\PyCharm\desktop_agent\stage1\img.png"},
            {"type": "text", "text": "罗列出图片中所有的应用"},
        ],
    }
]

start_time = time.time()
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

end_time = time.time()
print(f"推理时间：{end_time - start_time}秒")
print(output_text)