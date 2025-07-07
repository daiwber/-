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
# 对象识别：如 “图中有多少个图标？”、“图片中的主要应用是什么？”。
# 场景描述：如 “这个应用的界面布局是怎样的？”、“图中显示的是哪个功能模块？”。
# 操作状态：如 “当前应用处于什么状态？”、“用户正在进行什么操作？”。
# 文本信息：如 “图中有哪些文字内容？”、“界面中的按钮标签是什么？”。
# 综合理解：如 “根据界面信息推测用户的需求是什么？”。

# 在推理代码前添加计时开始
start_time = time.time()
# Preparation for inference
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

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

# 在推理代码后添加计时结束
end_time = time.time()
print(f"推理时间：{end_time - start_time}秒")
print(output_text)