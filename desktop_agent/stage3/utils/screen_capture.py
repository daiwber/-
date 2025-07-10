import mss
from mss.screenshot import ScreenShot
import numpy as np
from PIL import Image


def capture_screen(region=None):
    """捕获屏幕区域"""
    try:
        with mss.mss() as sct:
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