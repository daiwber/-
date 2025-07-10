import os
import re
import tempfile
import cv2
import numpy as np
from mss import mss
from core.task_state import TaskState
from core.operation import Operation
from models import load_model_and_processor
import time
import json
import threading
import queue
import pyautogui
from PIL import Image, ImageFont, ImageDraw
from core.task_prompts import task_prompts
from config import DEBUG_MODE, MAX_RETRIES, OPERATION_DELAY, VALID_OPERATIONS, SUPPORTED_APPS, SYSTEM_PROMPT, MODEL_PATH
from qwen_vl_utils import process_vision_info


model, processor = load_model_and_processor()

# 智能体核心类
class DesktopAutomationAgent:
    def __init__(self):
        self.task_name = ""
        self.target_app = ""
        self.username = ""
        self.password = ""
        self.search_term = ""
        self.operations = []
        self.current_step = 0
        self.state = TaskState.IDLE
        self.context = {}
        self.retry_count = 0
        self.last_screenshot = None
        self.operation_queue = queue.Queue()
        self.lock = threading.Lock()
        self.task_thread = None
        self.sct = mss()

    def capture_screen(self, region=None):
        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                if region:
                    monitor = {
                        "top": region[1],
                        "left": region[0],
                        "width": region[2] - region[0],
                        "height": region[3] - region[1],
                        "mon": 1
                    }

                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                return np.array(img)
        except Exception as e:
            print(f"屏幕捕获失败: {str(e)}")
            return np.zeros((500, 500, 3), dtype=np.uint8)

    def save_temp_image(self, image_np):
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return temp_path

    def get_next_action(self, image_np, instruction, context=""):
        image_path = self.save_temp_image(image_np)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": f"任务上下文：{context}\n当前指令：{instruction}\n"
                                "请严格按以下格式返回JSON：\n"
                                "```json\n"
                                "{\n"
                                '  "operation": "操作类型(CLICK|DOUBLE_CLICK|TYPE|ENTER|WAIT|SCROLL|HOTKEY|SEARCH)",\n'
                                '  "target": "目标元素描述",\n'
                                '  "value": "操作值(对于TYPE是文本，对于SCROLL是方向)",\n'
                                '  "bbox": [x1, y1, x2, y2],\n'
                                '  "reason": "操作原因"\n'
                                "}\n"
                                "```"
                    }
                ]
            }
        ]

        try:
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

            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if DEBUG_MODE:
                print(f"模型输出: {output_text[0]}")

            json_match = re.search(r'```json\s*({.*?})\s*```', output_text[0], re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                action_data = json.loads(json_str)

                op_type = action_data.get("operation", "").upper()
                if op_type not in VALID_OPERATIONS:
                    raise ValueError(f"无效的操作类型: {op_type}")

                action_data["operation"] = op_type
                return action_data
            else:
                raise ValueError("未找到有效的JSON响应")

        except Exception as e:
            print(f"获取操作时出错: {str(e)}")
            return {
                "operation": "WAIT",
                "target": "错误恢复",
                "value": "",
                "bbox": None,
                "reason": f"解析操作失败: {str(e)}"
            }
        finally:
            try:
                os.remove(image_path)
            except:
                pass

    def visualize_operation(self, image_np, operation, step_info):
        if image_np is None:
            return image_np

        img_pil = Image.fromarray(image_np)
        draw = ImageDraw.Draw(img_pil)

        try:
            font_large = ImageFont.truetype("arialbd.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        draw.rectangle([(10, 10), (400, 90)], fill=(0, 0, 0, 180))
        draw.text((20, 15), f"步骤: {step_info}", fill=(0, 255, 255), font=font_large)
        draw.text((20, 50), f"操作: {operation['operation']}", fill=(255, 255, 0), font=font_small)
        draw.text((20, 75), f"目标: {operation['target']}", fill=(255, 255, 0), font=font_small)

        if operation.get("bbox"):
            x1, y1, x2, y2 = operation["bbox"]
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            draw = ImageDraw.Draw(img_pil)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2], outline=(0, 200, 0), width=1)

            corner_size = 15
            draw.rectangle([x1, y1, x1 + corner_size, y1 + corner_size], fill=(255, 0, 255))
            draw.rectangle([x2 - corner_size, y1, x2, y1 + corner_size], fill=(255, 0, 255))
            draw.rectangle([x1, y2 - corner_size, x1 + corner_size, y2], fill=(255, 0, 255))
            draw.rectangle([x2 - corner_size, y2 - corner_size, x2, y2], fill=(255, 0, 255))

            pulse_radius = int(abs((time.time() % 1 - 0.5) * 20 + 15))
            draw.ellipse([center_x - pulse_radius, center_y - pulse_radius,
                          center_x + pulse_radius, center_y + pulse_radius],
                         outline=(255, 255, 0), width=3)

            arrow_size = 30
            draw.line([(center_x, center_y - arrow_size), (center_x, center_y + arrow_size)],
                      fill=(0, 255, 255), width=3)
            draw.line([(center_x - arrow_size, center_y), (center_x + arrow_size, center_y)],
                      fill=(0, 255, 255), width=3)

        return np.array(img_pil)

    def execute_operation(self, operation, context):
        op_type = operation["operation"].upper()
        target = operation["target"]
        value = operation.get("value", "")
        bbox = operation.get("bbox")

        # 获取屏幕尺寸
        screen_width, screen_height = pyautogui.size()

        # 实际执行操作
        try:
            if op_type == "CLICK" and bbox:
                # 确保坐标在屏幕范围内
                x1 = max(0, min(bbox[0], screen_width - 1))
                y1 = max(0, min(bbox[1], screen_height - 1))
                x2 = max(0, min(bbox[2], screen_width - 1))
                y2 = max(0, min(bbox[3], screen_height - 1))
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                pyautogui.click(center_x, center_y)
                return f"实际点击: {target} 在位置 ({center_x}, {center_y})"

            elif op_type == "DOUBLE_CLICK" and bbox:
                # 确保坐标在屏幕范围内
                x1 = max(0, min(bbox[0], screen_width - 1))
                y1 = max(0, min(bbox[1], screen_height - 1))
                x2 = max(0, min(bbox[2], screen_width - 1))
                y2 = max(0, min(bbox[3], screen_height - 1))
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                pyautogui.doubleClick(center_x, center_y)
                time.sleep(3)
                return f"实际双击: {target} 在位置 ({center_x}, {center_y})"

            elif op_type == "TYPE" and value and bbox:
                # 确保坐标在屏幕范围内
                x1 = max(0, min(bbox[0], screen_width - 1))
                y1 = max(0, min(bbox[1], screen_height - 1))
                x2 = max(0, min(bbox[2], screen_width - 1))
                y2 = max(0, min(bbox[3], screen_height - 1))
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                pyautogui.click(center_x, center_y)
                time.sleep(0.5)
                pyautogui.write(value)
                time.sleep(1)
                pyautogui.press('enter')
                return f"实际输入: '{value}' 到 {target}"

            elif op_type == "ENTER":
                pyautogui.press('enter')
                return f"实际按下回车键"

            elif op_type == "WAIT":
                try:
                    duration = float(value) if value else 2.0
                    time.sleep(min(duration, 10))
                    return f"实际等待: {duration}秒"
                except:
                    time.sleep(2)
                    return f"实际等待: 2秒"

            elif op_type == "SCROLL":
                direction = "up" if value.lower() in ["up", "u"] else "down"
                pyautogui.scroll(100 if direction == "up" else -100)
                return f"实际滚动: {'向上' if direction == 'up' else '向下'}"

            elif op_type == "HOTKEY":
                keys = value.split('+')
                pyautogui.hotkey(*keys)
                return f"实际快捷键: {value}"

            # 修改SEARCH操作处理
            elif op_type == "SEARCH":
                if not bbox:
                    raise ValueError("缺少搜索目标坐标")

                # 确保坐标在屏幕范围内
                x1 = max(0, min(bbox[0], screen_width - 1))
                y1 = max(0, min(bbox[1], screen_height - 1))
                x2 = max(0, min(bbox[2], screen_width - 1))
                y2 = max(0, min(bbox[3], screen_height - 1))

                # 计算输入框的中心位置
                input_x, input_y = (x1 + x2) // 2, (y1 + y2) // 2

                # 确保坐标有效
                if input_x < 0 or input_x >= screen_width or input_y < 0 or input_y >= screen_height:
                    raise ValueError(f"无效坐标: ({input_x}, {input_y})")

                # 点击搜索框
                pyautogui.click(input_x, input_y)
                time.sleep(0.5)

                # 清空现有内容
                pyautogui.hotkey('ctrl', 'a')
                pyautogui.press('backspace')
                time.sleep(0.3)

                # 输入关键词
                search_term = value if value else "默认搜索"
                pyautogui.write(search_term)
                time.sleep(0.5)

                # 按回车执行搜索
                pyautogui.press('enter')

                return f"执行搜索: '{search_term}'"

            # 如果没有任何匹配的操作，返回模拟操作
            return f"执行操作: {op_type} 目标: {target}"

        except Exception as e:
            print(f"操作执行错误: {str(e)}")
            raise  # 重新抛出异常以触发重试机制

    def run_task(self, task_name, app_name, username, password, search_term="", progress_callback=None,
                 status_callback=None):
        with self.lock:
            self.task_name = task_name
            self.target_app = app_name
            self.username = username
            self.password = password
            self.search_term = search_term
            self.operations = []
            self.current_step = 0
            self.state = TaskState.RUNNING
            self.retry_count = 0
            self.context = {
                "target_app": app_name,
                "username": username,
                "password": password,
                "search_term": search_term,
                "app_info": SUPPORTED_APPS.get(app_name, SUPPORTED_APPS["联想浏览器"])
            }

            task_instructions = task_prompts.get(task_name, [])
            if not task_instructions:
                self.state = TaskState.ERROR
                if status_callback:
                    status_callback("错误：未知任务")
                return

            if status_callback:
                status_callback(f"开始任务: {task_name} - 目标应用: {app_name}")
                status_callback("注意: 请确保目标应用在桌面上可见")

            while self.current_step < len(task_instructions) and self.state == TaskState.RUNNING:
                step_desc = f"{self.current_step + 1}/{len(task_instructions)}"
                instruction = task_instructions[self.current_step]

                # 动态替换指令中的占位符
                instruction = instruction.replace("{app_name}", app_name)
                instruction = instruction.replace("{app_icon}", self.context["app_info"]["icon"])

                # 登录任务相关替换
                if "login_window" in instruction:
                    instruction = instruction.replace("{login_window}",
                                                      self.context["app_info"].get("login_window", ""))

                if "{account_login_tab}" in instruction:
                    account_tab = self.context["app_info"].get("account_login_tab", "账密登录标签")
                    instruction = instruction.replace("{account_login_tab}", account_tab)

                instruction = instruction.replace("{username_field}",
                                                  self.context["app_info"].get("username_field", ""))
                instruction = instruction.replace("{password_field}",
                                                  self.context["app_info"].get("password_field", ""))
                instruction = instruction.replace("{login_button}", self.context["app_info"].get("login_button", ""))

                # 搜索任务相关替换
                instruction = instruction.replace("{address_bar}",
                                                  self.context["app_info"].get("address_bar", "地址栏"))
                instruction = instruction.replace("{search_box}", self.context["app_info"].get("search_box", "搜索框"))
                instruction = instruction.replace("{search_button}",
                                                  self.context["app_info"].get("search_button", "搜索按钮"))

                # 用户凭据替换
                instruction = instruction.replace("{username}", username)
                instruction = instruction.replace("{password}", password)
                instruction = instruction.replace("{search_term}", self.context.get("search_term", ""))

                if status_callback:
                    status_callback(f"步骤 {step_desc}: {instruction}")

                # 重置每个步骤的重试计数器
                step_retry_count = 0
                step_completed = False

                while not step_completed and step_retry_count <= MAX_RETRIES:
                    try:
                        screenshot = self.capture_screen()
                        self.last_screenshot = screenshot.copy()

                        context_info = f"任务: {task_name}, 目标应用: {app_name}, 步骤: {step_desc}, 历史: {self.context}"

                        operation_data = self.get_next_action(
                            screenshot,
                            instruction,
                            context_info
                        )

                        # 检查是否是错误恢复操作
                        if operation_data["operation"] == "WAIT" and "解析操作失败" in operation_data["reason"]:
                            raise ValueError(f"操作解析失败: {operation_data['reason']}")

                        visualized_img = self.visualize_operation(
                            screenshot,
                            operation_data,
                            step_desc
                        )

                        result = self.execute_operation(operation_data, self.context)

                        op = Operation(
                            op_type=operation_data["operation"],
                            target=operation_data["target"],
                            value=operation_data.get("value", ""),
                            coordinates=operation_data.get("bbox"),
                            description=operation_data["reason"],
                            vis_image=visualized_img
                        )
                        op.success = True
                        self.operations.append(op)

                        self.context[f"step_{self.current_step}"] = {
                            "operation": op.type,
                            "target": op.target,
                            "result": result
                        }

                        if progress_callback:
                            progress_callback(visualized_img, result)

                        step_completed = True
                        step_retry_count = 0
                        time.sleep(OPERATION_DELAY)

                    except Exception as e:
                        print(f"步骤执行出错: {str(e)}")
                        step_retry_count += 1
                        self.retry_count += 1

                        # 创建错误操作记录
                        error_op = Operation(
                            op_type="ERROR",
                            target="错误恢复",
                            value=str(e),
                            description=f"步骤失败: {str(e)}",
                            coordinates=None
                        )
                        error_op.success = False
                        self.operations.append(error_op)

                        if step_retry_count > MAX_RETRIES:
                            if status_callback:
                                status_callback(f"错误: 步骤 '{instruction}' 失败超过最大重试次数")
                            self.state = TaskState.ERROR
                            break

                        if status_callback:
                            status_callback(f"重试步骤 {step_desc} ({step_retry_count}/{MAX_RETRIES})")

                        # 等待时间随重试次数增加而增加
                        delay = OPERATION_DELAY * (step_retry_count + 1)
                        time.sleep(delay)

                # 如果步骤成功完成，进入下一步
                if step_completed:
                    self.current_step += 1
                    self.retry_count = 0
                else:
                    # 步骤失败，退出任务
                    break

            if self.state == TaskState.RUNNING:
                self.state = TaskState.COMPLETED
                if status_callback:
                    status_callback(f"任务 '{task_name}' 成功完成!")

    def start_task(self, task_name, app_name, username, password, search_term, progress_callback=None,
                   status_callback=None):
        if self.state == TaskState.RUNNING:
            if status_callback:
                status_callback("错误：已有任务正在运行")
            return

        self.task_thread = threading.Thread(
            target=self.run_task,
            args=(task_name, app_name, username, password, search_term, progress_callback, status_callback),
            daemon=True
        )
        self.task_thread.start()

    def pause_task(self):
        if self.state == TaskState.RUNNING:
            self.state = TaskState.PAUSED

    def resume_task(self):
        if self.state == TaskState.PAUSED:
            self.state = TaskState.RUNNING

    def stop_task(self):
        self.state = TaskState.IDLE

    def get_task_report(self):
        report = {
            "task_name": self.task_name,
            "target_app": self.target_app,
            "username": self.username,
            "status": self.state,
            "steps_completed": self.current_step,
            "total_steps": len(task_prompts.get(self.task_name, [])),
            "operations": [op.to_dict() for op in self.operations]
        }
        return json.dumps(report, indent=2)

    def get_current_state(self):
        return {
            "task_name": self.task_name,
            "target_app": self.target_app,
            "state": self.state,
            "current_step": self.current_step,
            "total_steps": len(task_prompts.get(self.task_name, [])),
            "retry_count": self.retry_count
        }