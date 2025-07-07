from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import BitsAndBytesConfig
import time
import cv2
import numpy as np
import re
import json
import os
os.environ["USE_FLASH_ATTENTION"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

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
)


def get_element_coordinates(image_path, instruction):
    """
    根据图像和自然语言指令，获取UI元素的边界框坐标
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]

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
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # 在推理代码后添加计时结束
    end_time = time.time()
    print(f"推理时间：{end_time - start_time}秒")
    print("模型输出:", output_text)

    # 解析JSON格式的输出
    try:
        # 假设输出是 ```json [ { "bbox_2d": [x1, y1, x2, y2], "label": "label" } ] ``` 格式
        json_str = output_text[0].strip('```json').strip()
        data = json.loads(json_str)

        if isinstance(data, list) and len(data) > 0:
            # 提取第一个对象的bbox_2d坐标
            coordinates = data[0].get("bbox_2d", None)
            if coordinates and len(coordinates) == 4:
                return coordinates
    except json.JSONDecodeError:
        print("模型输出不是有效的JSON格式")

    return None


def visualize_coordinates(image_path, coordinates):
    """
    在原图上可视化显示边界框
    """
    # 读取原图
    image = cv2.imread(image_path)

    if image is None:
        print("无法加载图像，请检查路径是否正确")
        return

    if coordinates:
        # 绘制边界框
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow("Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未找到有效坐标")


if __name__ == "__main__":
    image_path = "D:\project-file\PyCharm\zheruan\img.png"
    instruction = "请返回微信的坐标，格式为JSON"
    print("模型输入:",instruction)
    # 获取坐标
    coordinates = get_element_coordinates(image_path, instruction)

    if coordinates:
        # 可视化坐标
        visualize_coordinates(image_path, coordinates)
    else:
        print("未找到有效坐标")