from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np


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