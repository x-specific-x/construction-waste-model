from inference_sdk import InferenceHTTPClient
from PIL import Image
import base64

# 初始化 Roboflow 客户端
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="NospYypRZBtI13sAxp9E"  # 替换为你的实际 API Key
)

# 预设类别列表（你训练的10类）
known_categories = [
    "aggregate", "asphalt", "excavated_materials", "expanded_polystyrene", "glass",
    "metal", "plastic", "public_fill", "pulverized_fue_ash", "rubber"
]

# 推理图片路径
image_path = "D://000057.jpg"

# 调用 Roboflow 模型进行推理
result = CLIENT.infer(image_path, model_id="construction_waste_classification-2v2ay/1")

# 提取识别结果
predictions = result.get("predictions", [])

if predictions:
    top_class = predictions[0]["class"]
    confidence = predictions[0]["confidence"]

    print("Roboflow识别结果：", top_class, "置信度：", confidence)

    # 判断是否需要调用 GPT-4o-mini 进行补充识别
    if confidence < 0.6 or top_class not in known_categories:
        print("识别置信度低或类别不在预设列表，调用 GPT-4o-mini 进行补充识别...")

        # 将图片转为 base64（用于上传或传给 GPT）
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # 模拟 GPT-4o-mini 返回结构化语义描述（你可以替换为实际调用结果）
        gpt_description = {
            "summary": "图像显示为混合建筑废料，包含木材碎片、塑料包装和纸板。",
            "possible_materials": ["wood", "plastic", "cardboard"],
            "recyclability": "部分材料可回收，例如塑料和纸板，可用于再制造或包装。",
            "suggested_use": "建议进一步分类后用于家具制造或包装材料回收。",
            "confidence": "中等"
        }

        # 打印结构化语义描述
        print("GPT补充识别结果：")
        for key, value in gpt_description.items():
            print(f"{key}: {value}")
    else:
        print("识别结果在预设类别范围内，置信度足够，无需补充识别。")
else:
    print("未识别出任何类别")