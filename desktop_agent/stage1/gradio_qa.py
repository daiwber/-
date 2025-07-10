from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import BitsAndBytesConfig
import time
import gradio as gr
from PIL import Image
import os
import re
from system_prompt import prompt

if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

# 指定本地模型文件夹路径
model_path = "D:\Downloads\Model\Qwen2.5-VL-3B-Instruct"

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("正在加载模型...请耐心等待...")

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    local_files_only=True
)

print("正在加载处理器...")
# 加载处理器
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=True  # 强制使用快速处理器
)


def safe_image_path(path):
    """安全处理Windows路径格式问题"""
    if path.startswith('file=') or '?' in path:
        # 处理Gradio上传的临时文件格式
        return re.sub(r"^file=(.*?)(\?.*)?$", r"\1", path)
    return path


def validate_image(image_path):
    """验证图片路径是否有效"""
    if not image_path:
        return False, "请上传图片"

    clean_path = safe_image_path(image_path)

    if not os.path.exists(clean_path):
        return False, f"图片路径不存在: {clean_path}"

    try:
        img = Image.open(clean_path)
        img.verify()  # 验证是否为有效图片
        img.close()  # 关闭文件
        return True, clean_path
    except Exception as e:
        return False, f"无效图片文件: {str(e)}"


def generate_response(image_path, question):
    """生成模型响应"""
    # 验证并清理图片路径
    is_valid, validated_path = validate_image(image_path)
    if not is_valid:
        return validated_path, "0.00秒", validated_path

    # 准备消息
    messages = [
        {
            "role": "system",  # 系统级指令
            "content": prompt
        },
        {
        "role": "user",
        "content": [
            {"type": "image", "image": validated_path},
            {"type": "text", "text": question},
        ],
    }]

    start_time = time.time()

    try:
        # 准备输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)  # 忽略视频输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 生成响应
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # 解码响应
        decoded_output = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 清理输出文本
        output_text = decoded_output.replace("<|im_end|>", "").strip()
        print(output_text)
        elapsed = time.time() - start_time
        return output_text, f"{elapsed:.2f}秒", ""
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return "", "0.00秒", f"推理错误: {str(e)}\n详细信息: {tb}"


# 创建Gradio界面
with gr.Blocks(title="Qwen2.5-VL多模态模型", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Qwen2.5-VL多模态模型演示")
    gr.Markdown("上传图片并向模型提问，模型将分析图片内容并回答您的问题")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="上传图片",
                height=300,
                interactive=True,
                sources=["upload", "clipboard"]
            )

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="您的问题",
                placeholder="例如：图片中有哪些应用？图中显示了什么内容？这些图标的含义是什么？",
                lines=3
            )
            with gr.Row():
                submit_btn = gr.Button("提交问题", variant="primary")
                clear_btn = gr.Button("清空内容", variant="secondary")

    # 示例区域
    with gr.Accordion("点击查看示例问题", open=False):
        with gr.Row():
            gr.Examples(
                examples=[["D:\project-file\PyCharm\desktop_agent\stage1\img.png", "罗列出图片中所有的应用"]],
                inputs=[image_input, question_input],
                label="完整示例"
            )

            gr.Examples(
                examples=[
                    ["图中有多少个图标？"],
                    ["图片中的主要应用是什么？"],
                    ["图片背景是什么颜色？"],
                    ["图中有哪些文字内容？"],
                    ["界面中哪个应用最引人注目？"]
                ],
                inputs=[question_input],
                label="问题示例"
            )

    # 结果区域
    with gr.Row():
        output_text = gr.Textbox(
            label="模型回答",
            placeholder="等待回答...",
            lines=6,
            interactive=False,
            show_copy_button=True
        )

    with gr.Row():
        time_output = gr.Textbox(
            label="推理耗时",
            interactive=False,
            scale=1
        )
        gr.Textbox(
            label="状态",
            value="就绪",
            interactive=False,
            scale=1
        )
        info_output = gr.Textbox(
            label="系统信息",
            value=f"PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}",
            interactive=False,
            scale=2
        )

    error_output = gr.Textbox(
        label="错误信息",
        visible=False,
        interactive=False
    )

    # 提交处理
    submit_btn.click(
        fn=generate_response,
        inputs=[image_input, question_input],
        outputs=[output_text, time_output, error_output]
    )


    # 清空按钮
    def clear_all():
        return [None, "", "", "0.00秒", ""]


    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, question_input, output_text, time_output, error_output]
    )

    # 输入变化时清空结果
    image_input.change(
        fn=lambda: ["", "0.00秒", ""],
        outputs=[output_text, time_output, error_output]
    )
    question_input.change(
        fn=lambda: ["", "0.00秒", ""],
        outputs=[output_text, time_output, error_output]
    )

# 启动界面
if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        favicon_path=None,
        show_error=True
    )