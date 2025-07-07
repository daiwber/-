import tempfile
import os
import time

import cv2
import numpy as np

def save_temp_image(image_np):
    """保存临时图像文件"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return temp_path