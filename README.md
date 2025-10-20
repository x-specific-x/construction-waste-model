# construction_waste_classification 建筑废料图像识别模块（用于智能配对平台）

## 项目背景

本模块是建筑废料智能配对平台的一部分，旨在识别上传的建筑废料图像是否属于可回收类别，并为后续的语义匹配与资源配对提供结构化信息。

参考香港环保署的建筑废料回收类别（[官方链接]([https://www.epd.gov.hk*10 类可回收建筑废料](https://www.epd.gov.hk/epd/misc/cdm/b5_products1.htm)：

| 中文类别 | 英文类别（模型标签） |
|----------|----------------------|
| 玻璃 | `glass` |
| 公众填料（瓦砾及混凝土） | `public_fill` |
| 金属 | `metal` |
| 沥青 | `asphalt` |
| 煤灰 | `pulverized_fue_ash` |
| 泡沫塑料 | `expanded_polystyrene` |
| 塑胶 | `plastic` |
| 碎石骨料 | `aggregate` |
| 挖掘料 | `excavated_materials` |
| 橡胶 | `rubber` |

每类爬取约 50 张图像，并上传至 Roboflow 平台进行训练，构建了一个轻量级图像分类模型。

---

## 模型训练流程（使用 Roboflow）

1. 注册并登录 [Roboflow](https://roboflow.com)
3. 上传图像文件夹（每个子文件夹代表一个类别）
4. 生成数据版本（可选择增强）
5. 点击 **Train Model** 开始训练
6. 获取模型 ID（如 `your_model_id/1`）用于推理

> 注意：Roboflow 上传时会自动过滤部分图像，所以最终我的classification_model文件是在项目页面点击 **Download**，选择 `Download zip to computer` 下载已处理的数据集。

---

## 如何运行classification.py文件

建议使用 **Python 3.10** 环境，并通过 `conda` 安装依赖：

```bash
conda create -n your_env python=3.10
conda activate your_env
pip install git+https://github.com/roboflow/inference.git
```
