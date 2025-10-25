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
        
    else:
        print("识别结果在预设类别范围内，置信度足够，无需补充识别。")
else:
    print("未识别出任何类别")
