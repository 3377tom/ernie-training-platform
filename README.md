# ERNIE 3.0 Mini 训练评估平台

一个完整的数据标注格式与训练-评估-压缩-ONNX转换的平台，基于ERNIE 3.0 Mini模型。

## 目录结构

```
ERNIE_Training_Platform/
├── data/                   # 数据集目录
│   ├── raw/               # 原始数据
│   ├── annotated/         # 标注后的数据
│   └── processed/         # 处理后的数据
├── scripts/               # 脚本目录
│   ├── data_annotation.py  # 数据标注脚本
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── compress.py         # 模型压缩脚本
│   └── convert_to_onnx.py  # ONNX转换脚本
├── configs/               # 配置文件
│   ├── model_config.json   # 模型配置
│   └── training_config.json # 训练配置
├── models/                # 模型目录
│   ├── pretrained/         # 预训练模型
│   ├── trained/            # 训练后的模型
│   ├── compressed/         # 压缩后的模型
│   └── onnx/               # ONNX模型
├── utils/                 # 工具函数
│   ├── data_processor.py   # 数据处理工具
│   ├── metrics.py          # 评估指标
│   └── logger.py           # 日志工具
└── requirements.txt        # 依赖文件
```

## 数据标注格式

使用JSON格式进行数据标注，示例如下：

```json
{
  "id": "sample_001",
  "text": "这是一个示例文本",
  "label": "positive",
  "meta": {
    "source": "example",
    "timestamp": "2024-01-01 12:00:00"
  }
}
```

## 环境准备

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据标注

```bash
python scripts/data_annotation.py --input data/raw --output data/annotated
```

### 2. 模型训练

```bash
python scripts/train.py --config configs/training_config.json
```

### 3. 模型评估

```bash
python scripts/evaluate.py --model models/trained --dataset data/processed/test
```

### 4. 模型压缩

```bash
python scripts/compress.py --model models/trained --output models/compressed
```

### 5. ONNX转换

```bash
python scripts/convert_to_onnx.py --model models/compressed --output models/onnx
```

## 配置文件说明

### training_config.json

```json
{
  "model_name": "ernie-3.0-mini-zh",
  "task_type": "classification",
  "num_classes": 2,
  "train_data": "data/processed/train",
  "val_data": "data/processed/val",
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 2e-5,
  "max_seq_length": 128,
  "save_steps": 1000,
  "output_dir": "models/trained"
}
```

## 支持的任务类型

- 文本分类
- 序列标注
- 问答任务
- 文本生成

## 模型压缩策略

- 知识蒸馏
- 模型剪枝
- 量化

## ONNX转换

支持将训练好的ERNIE模型转换为ONNX格式，方便部署到不同平台。

## 许可证

Apache License 2.0"# ernie-training-platform" 
