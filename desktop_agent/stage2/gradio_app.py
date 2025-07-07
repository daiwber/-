import gradio as gr
import os
import cv2
import numpy as np
import time
import json
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import BitsAndBytesConfig
import warnings
import tempfile
from agent_prompt import ui_prompt

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# 指定本地模型文件夹路径
model_path = "D:\\Downloads\\Model\\Qwen2.5-VL-3B-Instruct"

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 加载模型时指定local_files_only=True，并应用量化配置
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

# 加载分词器时指定local_files_only=True
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True
)


def save_temp_image(image_np):
    """将numpy数组图像保存为临时文件并返回路径"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return temp_path


def get_element_coordinates(image_np, instruction):
    """
    根据图像和自然语言指令，获取UI元素的边界框坐标
    """
    # 将numpy数组保存为临时文件
    image_path = save_temp_image(image_np)

    messages = [
        {
            "role": "system",  # 系统级指令
            "content": ui_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": instruction,
                },
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

    # 清理临时文件
    try:
        os.remove(image_path)
    except:
        pass

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


def visualize_coordinates(image_np, coordinates):
    """
    在原图上可视化显示边界框（炫酷版）
    """
    if image_np is None:
        print("无法加载图像，请检查路径是否正确")
        return None

    if not coordinates:
        print("未找到有效坐标")
        return image_np

    # 创建副本
    image = image_np.copy()
    x1, y1, x2, y2 = coordinates

    # ========== 1. 动态发光边框 ==========
    border_color = (0, 255, 255)  # 青色
    thickness = 3
    glow_size = 5

    # 发光效果（多层模糊边框）
    for i in range(glow_size, 0, -1):
        alpha = i / glow_size * 0.3  # 透明度递减
        overlay = image.copy()
        cv2.rectangle(overlay, (x1 - i, y1 - i), (x2 + i, y2 + i), border_color, thickness)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # ========== 2. 3D立体边框 ==========
    # 上边和左边用亮色
    cv2.line(image, (x1, y1), (x2, y1), (100, 255, 100), thickness)
    cv2.line(image, (x1, y1), (x1, y2), (100, 255, 100), thickness)
    # 下边和右边用暗色
    cv2.line(image, (x1, y2), (x2, y2), (0, 150, 0), thickness)
    cv2.line(image, (x2, y1), (x2, y2), (0, 150, 0), thickness)

    # ========== 3. 动态标记点 ==========
    corner_color = (255, 0, 255)  # 品红色
    corner_size = 10
    # 四个角画圆
    cv2.circle(image, (x1, y1), corner_size, corner_color, -1)
    cv2.circle(image, (x1, y2), corner_size, corner_color, -1)
    cv2.circle(image, (x2, y1), corner_size, corner_color, -1)
    cv2.circle(image, (x2, y2), corner_size, corner_color, -1)

    # ========== 4. 中心标记 + 动画效果 ==========
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    # 脉冲圆环动画效果
    pulse_radius = int(abs((time.time() % 1 - 0.5) * 20 + 15)) # 15-35之间脉动
    cv2.circle(image, (center_x, center_y), pulse_radius, (255, 255, 0), 2)

    # ========== 5. 信息标签 ==========
    label = "DETECTED!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
    # 文字背景
    cv2.rectangle(image, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), (50, 50, 220), -1)
    # 文字
    cv2.putText(image, label, (x1 + 5, y1 - 10), font, 0.7, (255, 255, 255), 2)

    # ========== 6. 区域高亮 ==========
    highlight = image.copy()
    mask = np.zeros_like(highlight)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    highlight = cv2.bitwise_and(highlight, mask)
    image = cv2.addWeighted(image, 0.7, highlight, 0.3, 0)

    return image


def process_ui_detection(image_np, instruction):
    """
    处理UI检测
    """
    coordinates = get_element_coordinates(image_np, instruction)
    result_image = visualize_coordinates(image_np, coordinates)
    return result_image, json.dumps({"coordinates": coordinates}) if coordinates else "未找到有效坐标"


# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🖼️ UI元素检测系统
    使用Qwen2.5-VL多模态模型根据自然语言指令定位UI元素
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="上传界面截图", type="numpy")
            instruction_input = gr.Textbox(
                label="检测指令",
                placeholder="例如：'请返回微信图标的坐标'",
                lines=2
            )
            run_btn = gr.Button("检测元素", variant="primary")

            gr.Examples(
                examples=[
                    [os.path.join(os.path.dirname(__file__), "img.png"), "请寻找Cursor"],
                    [os.path.join(os.path.dirname(__file__), "img.png"), "请寻找微信"]
                ],
                inputs=[image_input, instruction_input],
                label="示例输入"
            )

        with gr.Column():
            image_output = gr.Image(label="检测结果", interactive=False)
            text_output = gr.Textbox(label="检测信息", interactive=False)

    run_btn.click(
        fn=process_ui_detection,
        inputs=[image_input, instruction_input],
        outputs=[image_output, text_output]
    )

    gr.Markdown("""
### 使用说明：
1. 上传界面截图
2. 输入自然语言指令描述要定位的元素
3. 点击"检测元素"按钮
4. 结果将显示检测框和坐标信息

**注意**：首次运行需要加载模型，可能需要较长时间
""")

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )