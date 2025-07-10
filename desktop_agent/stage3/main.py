import gradio as gr
# main.py - 主程序入口
from interface.gradio_ui import demo

if __name__ == "__main__":
    try:
        demo.queue(concurrency_count=5)
        demo.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            show_error=True,
            enable_queue=True
        )
    except Exception as e:
        print(f"应用启动失败: {str(e)}")
        with gr.Blocks() as error_demo:
            gr.Markdown(f"# 应用启动失败")
            gr.Markdown(f"错误信息: {str(e)}")
            gr.Markdown("请检查控制台日志获取更多信息")
        error_demo.launch()