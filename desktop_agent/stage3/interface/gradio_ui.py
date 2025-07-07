import gradio as gr
import numpy as np

from agent.desktop_automation import DesktopAutomationAgent
from config import SUPPORTED_APPS, DEBUG_MODE
from core.task_prompts import task_prompts
import re
import time

from core.task_state import TaskState

# 创建智能体实例

# Gradio界面代码，与原始代码相同
# 创建智能体实例
agent = DesktopAutomationAgent()

# Gradio界面
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="amber")) as demo:
    # 状态变量
    current_image = gr.State()
    task_history = gr.State([])
    selected_operation_index = gr.State(-1)
    operation_list = gr.State([])
    user_selected_index = gr.State(-1)
    manual_selection = gr.State(False)

    gr.Markdown("""
    # 🚀 桌面自动化智能体系统
    **支持用户登录和网页搜索功能**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                label="选择任务",
                choices=list(task_prompts.keys()),
                value="用户登录"
            )

            app_dropdown = gr.Dropdown(
                label="选择应用",
                choices=list(SUPPORTED_APPS.keys()),
                value="QQ"
            )

            username_input = gr.Textbox(
                label="用户名",
                placeholder="输入登录用户名",
                value="testuser",
                visible=True
            )

            password_input = gr.Textbox(
                label="密码",
                placeholder="输入登录密码",
                value="password123",
                type="password",
                visible=True
            )

            search_input = gr.Textbox(
                label="搜索关键词",
                placeholder="输入搜索内容",
                value="人工智能",
                visible=False
            )

            with gr.Row():
                start_btn = gr.Button("开始任务", variant="primary")
                pause_btn = gr.Button("暂停任务")
                resume_btn = gr.Button("继续任务")
                stop_btn = gr.Button("停止任务")

            status_display = gr.Textbox(
                label="任务状态",
                value="就绪",
                interactive=False
            )

            progress_bar = gr.Slider(
                label="任务进度",
                minimum=0,
                maximum=100,
                value=0,
                interactive=False
            )

            gr.Markdown("### 任务历史")
            history_display = gr.JSON(label="操作记录")

        with gr.Column(scale=2):
            image_output = gr.Image(
                label="屏幕监控",
                interactive=False,
                height=500
            )

            with gr.Accordion("操作详情", open=True):
                operation_output = gr.Textbox(
                    label="当前操作",
                    value="等待任务开始...",
                    interactive=False
                )

            with gr.Accordion("操作历史查看器", open=True):
                with gr.Row():
                    operation_selector = gr.Dropdown(
                        label="选择操作步骤",
                        choices=[],
                        interactive=True
                    )
                    refresh_btn = gr.Button("刷新", variant="secondary")

                operation_vis = gr.Image(
                    label="操作可视化",
                    interactive=False,
                    height=400
                )
                operation_details = gr.JSON(
                    label="操作详情"
                )

            with gr.Accordion("高级选项", open=False):
                with gr.Row():
                    capture_btn = gr.Button("捕获当前屏幕")
                    debug_toggle = gr.Checkbox(label="调试模式", value=DEBUG_MODE)

                report_btn = gr.Button("生成任务报告")
                report_output = gr.JSON(label="任务报告")


    # 回调函数
    def update_progress(image, result):
        return image, result


    def update_status(message):
        return message


    def start_task(task_name, app_name, username, password, search_term, history):
        if agent.state != TaskState.RUNNING:
            agent.start_task(
                task_name,
                app_name,
                username,
                password,
                search_term,
                progress_callback=update_progress,
                status_callback=update_status
            )
            return f"任务启动中...目标应用: {app_name}", False
        return "任务已在运行中", False


    def pause_task():
        agent.pause_task()
        return "任务已暂停"


    def resume_task():
        agent.resume_task()
        return "任务继续运行"


    def stop_task():
        agent.stop_task()
        return "任务已停止"


    def capture_screen():
        screenshot = agent.capture_screen()
        return screenshot


    def generate_report():
        return agent.get_task_report()


    def update_operation_selector():
        operations = agent.operations
        choices = [f"步骤 {i + 1}: {op.type} - {op.target}" for i, op in enumerate(operations)]
        return gr.Dropdown.update(choices=choices, value=choices[-1] if choices else None)


    def select_operation(operation_name, selected_index, user_selected_index, manual_selection):
        if not operation_name:
            return None, None, selected_index, user_selected_index, True

        match = re.search(r'步骤 (\d+):', operation_name)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(agent.operations):
                op = agent.operations[index]
                vis_image = op.vis_image
                details = {
                    "操作类型": op.type,
                    "目标元素": op.target,
                    "操作值": op.value if op.value else "无",
                    "坐标": op.coordinates if op.coordinates else "无",
                    "原因": op.description,
                    "时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(op.timestamp)),
                    "状态": "成功" if op.success else "失败"
                }
                return vis_image, details, index, index, True

        return None, None, selected_index, user_selected_index, manual_selection


    def update_ui(selected_index, user_selected_index, manual_selection):
        state = agent.get_current_state()

        if manual_selection and user_selected_index >= 0 and user_selected_index < len(agent.operations):
            selected_index = user_selected_index
        else:
            selected_index = len(agent.operations) - 1 if agent.operations else -1
            user_selected_index = selected_index

        total_steps = state["total_steps"] or 1
        progress = int((state["current_step"] / total_steps) * 100) if total_steps > 0 else 0
        history = [op.to_dict() for op in agent.operations]
        selector_choices = [f"步骤 {i + 1}: {op.type} - {op.target}" for i, op in enumerate(agent.operations)]

        if selected_index >= 0 and selected_index < len(selector_choices):
            current_value = selector_choices[selected_index]
        else:
            current_value = selector_choices[-1] if selector_choices else None
            selected_index = len(selector_choices) - 1 if selector_choices else -1

        if history:
            if selected_index >= 0 and selected_index < len(history):
                current_op = history[selected_index]["description"]
            else:
                current_op = history[-1]["description"]
        else:
            current_op = "等待操作..."

        if selected_index >= 0 and selected_index < len(agent.operations):
            vis_image = agent.operations[selected_index].vis_image
        else:
            vis_image = agent.last_screenshot if agent.last_screenshot is not None else np.zeros((100, 100, 3),
                                                                                                 dtype=np.uint8)

        return (
            state.get("state", "就绪"),
            progress,
            history,
            vis_image,
            current_op,
            gr.Dropdown.update(choices=selector_choices, value=current_value),
            selected_index,
            user_selected_index,
            manual_selection
        )


    def toggle_inputs(task_name):
        if task_name == "用户登录":
            return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
        elif task_name == "网页搜索":
            return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
        else:
            return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]


    # 事件绑定
    start_btn.click(
        start_task,
        inputs=[task_dropdown, app_dropdown, username_input, password_input, search_input],
        outputs=[status_display, manual_selection]
    )

    pause_btn.click(
        pause_task,
        outputs=[status_display]
    )

    resume_btn.click(
        resume_task,
        outputs=[status_display]
    )

    stop_btn.click(
        stop_task,
        outputs=[status_display]
    )

    capture_btn.click(
        capture_screen,
        outputs=[image_output]
    )

    report_btn.click(
        generate_report,
        outputs=[report_output]
    )

    operation_selector.change(
        select_operation,
        inputs=[operation_selector, selected_operation_index, user_selected_index, manual_selection],
        outputs=[operation_vis, operation_details, selected_operation_index, user_selected_index, manual_selection]
    )

    refresh_btn.click(
        lambda: (-1, -1, False),
        outputs=[selected_operation_index, user_selected_index, manual_selection]
    )

    task_dropdown.change(
        toggle_inputs,
        inputs=task_dropdown,
        outputs=[username_input, password_input, search_input]
    )

    demo.load(
        update_ui,
        inputs=[selected_operation_index, user_selected_index, manual_selection],
        outputs=[
            status_display,
            progress_bar,
            history_display,
            image_output,
            operation_output,
            operation_selector,
            selected_operation_index,
            user_selected_index,
            manual_selection
        ],
        every=0.5
    )

    gr.Markdown("""
    ## 功能说明

    **用户登录流程**:
    1. 选择"用户登录"任务
    2. 选择应用（QQ/微信/钉钉）
    3. 输入用户名和密码
    4. 点击"开始任务"按钮
    5. 观察自动化登录过程

    **网页搜索流程**:
    1. 选择"网页搜索"任务
    2. 选择浏览器（夸克/Edge/Firefox）
    3. 输入搜索关键词
    4. 点击"开始任务"按钮
    5. 观察自动化搜索过程

    **注意事项**:
    1. 确保目标应用在桌面上可见
    2. 保持应用为默认窗口大小
    3. 避免在自动化过程中操作鼠标键盘
    """)
