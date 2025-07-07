import torch
from transformers import BitsAndBytesConfig

DEBUG_MODE = True
MAX_RETRIES = 3
OPERATION_DELAY = 1.5  # 模拟操作延迟

# 指定本地模型文件夹路径
MODEL_PATH = "D:\\Downloads\\Model\\Qwen2.5-VL-3B-Instruct"

# 4-bit量化配置（移动到config.py）
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 支持的应用列表（更新了浏览器配置）
SUPPORTED_APPS = {
    "QQ": {
        "icon": "qq_极速版.png",
        "login_window": "QQ登录窗口",
        "username_field": "QQ号码输入框",
        "password_field": "QQ密码输入框",
        "login_button": "登录按钮",
        "account_login_tab": "账密登录标签"
    },
    "微信": {
        "icon": "wechat_icon.png",
        "login_window": "微信登录窗口",
        "username_field": "微信号输入框",
        "password_field": "微信密码输入框",
        "login_button": "登录按钮"
    },
    "钉钉": {
        "icon": "dingtalk_icon.png",
        "login_window": "钉钉登录窗口",
        "username_field": "手机号输入框",
        "password_field": "密码输入框",
        "login_button": "登录按钮"
    },
    "联想浏览器": {
        "icon": "D:\project-file\PyCharm\zheruan\img_1.png",
        "address_bar": "地址栏",
        "search_box": "搜索框",
        "search_button": "搜索按钮"
    },
    "Edge": {
        "icon": "edge_icon.png",
        "address_bar": "地址栏",
        "search_box": "搜索框",
        "search_button": "搜索按钮"
    },
    "Firefox": {
        "icon": "firefox_icon.png",
        "address_bar": "地址栏",
        "search_box": "搜索框",
        "search_button": "搜索按钮"
    },
    "自定义应用": {
        "icon": "default_icon.png",
        "login_window": "登录窗口",
        "username_field": "用户名输入框",
        "password_field": "密码输入框",
        "login_button": "登录按钮"
    }
}

SYSTEM_PROMPT = """
你是一个桌面自动化智能体，能够通过视觉理解桌面环境并执行操作。请根据提供的屏幕截图和用户指令，确定下一步操作。
操作类型包括：
- CLICK: 单击指定元素
- DOUBLE_CLICK: 双击指定元素
- TYPE: 在指定元素中输入文本
- ENTER: 按下回车键
- WAIT: 等待指定时间（秒）
- SCROLL: 滚动页面（上/下）
- HOTKEY: 执行组合快捷键
- SEARCH: 在搜索引擎中执行搜索

返回格式必须是严格的JSON格式，包含以下字段：
- operation: 操作类型（必须大写）
- target: 目标元素描述
- value: 操作值（对TYPE是文本，对WAIT是秒数，对SCROLL是方向）
- bbox: 目标元素的边界框 [x1, y1, x2, y2]
- reason: 操作原因解释

对于网页搜索任务，重点识别以下元素：
1. 浏览器地址栏：通常位于顶部，包含URL
2. 搜索框：页面中间或顶部的输入框
3. 搜索按钮：通常标有"搜索"、"百度一下"或放大镜图标
4. 应用图标，通常位于桌面左部，一般需要通过双击打开
"""



# 有效的操作类型
VALID_OPERATIONS = [
    "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK",
    "TYPE", "ENTER", "WAIT", "SCROLL", "DRAG", "HOTKEY", "SEARCH"
]