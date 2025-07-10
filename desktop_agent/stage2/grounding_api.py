import re

from openai import OpenAI
import base64
import cv2
import json
import os
import time


# 将图像文件转换为base64编码
def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_encoded = base64.b64encode(file_content)
        return base64_encoded.decode('utf-8')

# 使用 MindCraft API 获取元素坐标
def get_element_coordinates(image_path, instruction):
    # 配置 API 参数
    base_url = 'https://api.mindcraft.com.cn/v1/'
    api_key = 'MC-6451E2785EB4XXXXXXX'

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 构建 API 请求
    params = {
        "model": "GLM-4V-Flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    # 使用 base64 编码传输图像
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': file_to_base64(image_path)
                        },
                    },
                    {
                        'type': 'text',
                        'text': f"{instruction} 请以JSON格式返回结果，包含一个'bbox_2d'字段，值为[x1, y1, x2, y2],对应目标元素在原图中的坐标",
                    },
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
        "stream": False  # 使用非流式响应以便获取完整结果
    }

    try:
        # 记录开始时间
        start_time = time.time()

        # 发送 API 请求
        response = client.chat.completions.create(**params)

        # 获取响应内容
        content = response.choices[0].message.content

        # 记录结束时间
        end_time = time.time()
        print(f"API 调用时间：{end_time - start_time:.2f}秒")
        print("API 响应:", content)

        # 尝试解析 JSON 响应
        try:
            # 提取 JSON 部分
            json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            # 解析 JSON
            data = json.loads(content)

            if "bbox_2d" in data and isinstance(data["bbox_2d"], list) and len(data["bbox_2d"]) == 4:
                coordinates = [int(coord) for coord in data["bbox_2d"]]
                return coordinates
        except json.JSONDecodeError:
            print("响应不是有效的JSON格式")
            print(f"原始响应内容: {content}")

        return None

    except Exception as e:
        print(f"API 调用失败: {str(e)}")
        return None


# 在原图上可视化显示边界框
def visualize_coordinates(image_path, coordinates):
    # 读取原图
    image = cv2.imread(image_path)

    if image is None:
        print("无法加载图像，请检查路径是否正确")
        return

    if coordinates:
        # 绘制边界框
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加标签
        label = f"Target: ({x1}, {y1}) to ({x2}, {y2})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow("Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图像
        output_path = os.path.splitext(image_path)[0] + "_result.jpg"
        cv2.imwrite(output_path, image)
        print(f"结果已保存至: {output_path}")
    else:
        print("未找到有效坐标")


if __name__ == "__main__":
    image_path = "D:\project-file\PyCharm\desktop_agent\stage2\img.png"
    instruction = "请返回微信的坐标"

    print(f"图像路径: {image_path}")
    print(f"指令: {instruction}")

    # 获取坐标
    coordinates = get_element_coordinates(image_path, instruction)

    if coordinates:
        print(f"获取到坐标: {coordinates}")
        # 可视化坐标
        visualize_coordinates(image_path, coordinates)
    else:
        print("未找到有效坐标")