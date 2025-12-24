import click
import os
import sys
import json
import paddle
import onnx
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import global_logger as logger


@click.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help='训练好的模型路径')
@click.option('--output', '-o', type=click.Path(), required=True, help='ONNX模型输出路径')
@click.option('--opset-version', '-v', type=int, default=13, help='ONNX Opset版本')
@click.option('--batch-size', '-b', type=int, default=1, help='导出的ONNX模型的批量大小')
@click.option('--device', '-dev', type=str, default=None, help='指定运行设备')
def main(model: str, output: str, opset_version: int, batch_size: int, device: str):
    """ERNIE 3.0 Mini 模型转换为ONNX格式脚本"""
    # 1. 设置设备
    if device:
        paddle.set_device(device)
    else:
        device = paddle.get_device()
    logger.info(f"使用设备: {device}")
    
    # 2. 加载模型和分词器
    logger.info(f"加载模型: {model}")
    tokenizer = ErnieTokenizer.from_pretrained(model)
    
    # 加载标签映射
    label_mapping_path = os.path.join(model, 'label_mapping.json')
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        num_classes = len(label_mapping)
        logger.info(f"标签映射: {label_mapping}, 类别数: {num_classes}")
    else:
        logger.warning(f"未找到标签映射文件: {label_mapping_path}")
        num_classes = 2  # 默认二分类
    
    # 加载模型
    model = ErnieForSequenceClassification.from_pretrained(model, num_classes=num_classes)
    model.eval()
    
    # 3. 准备导出配置
    logger.info(f"准备导出ONNX模型，Opset版本: {opset_version}, 批量大小: {batch_size}")
    
    # 4. 构建输入张量
    max_seq_length = 128  # 默认序列长度
    
    # 动态轴配置，支持动态批量大小
    dynamic_axes = {
        'input_ids': {0: 'batch_size'},  # 输入的第一个维度是批量大小，设为动态
        'token_type_ids': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # 示例输入
    input_ids = paddle.ones(shape=[batch_size, max_seq_length], dtype='int64')
    token_type_ids = paddle.ones(shape=[batch_size, max_seq_length], dtype='int64')
    
    # 5. 导出ONNX模型
    os.makedirs(output, exist_ok=True)
    onnx_model_path = os.path.join(output, 'ernie_3.0_mini.onnx')
    
    logger.info(f"导出ONNX模型到: {onnx_model_path}")
    
    paddle.onnx.export(
        model,
        onnx_model_path.replace('.onnx', ''),  # paddle.onnx.export会自动添加.onnx后缀
        input_spec=[input_ids, token_type_ids],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        enable_onnx_checker=True
    )
    
    # 6. 验证ONNX模型
    logger.info("验证ONNX模型...")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX模型验证通过")
    
    # 7. 保存分词器和配置
    logger.info("保存分词器和配置文件...")
    
    # 保存分词器
    tokenizer.save_pretrained(output)
    
    # 复制标签映射
    if os.path.exists(label_mapping_path):
        import shutil
        shutil.copy2(label_mapping_path, os.path.join(output, 'label_mapping.json'))
    
    # 保存ONNX转换配置
    onnx_config = {
        'model_name': 'ernie-3.0-mini',
        'onnx_model_path': onnx_model_path,
        'opset_version': opset_version,
        'batch_size': batch_size,
        'max_seq_length': max_seq_length,
        'export_time': paddle.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    onnx_config_path = os.path.join(output, 'onnx_config.json')
    with open(onnx_config_path, 'w', encoding='utf-8') as f:
        json.dump(onnx_config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"ONNX转换配置已保存到: {onnx_config_path}")
    
    # 8. 生成使用示例
    generate_onnx_example(output)
    
    logger.info("ONNX模型转换完成！")
    logger.info(f"ONNX模型路径: {onnx_model_path}")
    logger.info(f"输出目录: {output}")


def generate_onnx_example(output_dir: str):
    """生成ONNX模型使用示例"""
    example_code = '''
# ERNIE 3.0 Mini ONNX模型使用示例

import onnxruntime as ort
from paddlenlp.transformers import ErnieTokenizer

# 加载分词器和ONNX模型
tokenizer = ErnieTokenizer.from_pretrained('{output_dir}')
sess = ort.InferenceSession('{output_dir}/ernie_3.0_mini.onnx')

# 文本处理
text = "这是一个测试文本"
inputs = tokenizer(
    text,
    max_seq_len=128,
    padding=True,
    truncation=True,
    return_token_type_ids=True
)

# 模型推理
input_ids = inputs['input_ids'].unsqueeze(0)  # 添加批量维度
token_type_ids = inputs['token_type_ids'].unsqueeze(0)

# ONNX推理
outputs = sess.run(
    None,
    {
        'input_ids': input_ids.numpy(),
        'token_type_ids': token_type_ids.numpy()
    }
)

# 获取预测结果
logits = outputs[0]
prediction = logits.argmax(axis=1)
print(f"预测结果: {prediction}")
'''.format(output_dir=output_dir)
    
    example_path = os.path.join(output_dir, 'onnx_inference_example.py')
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    logger.info(f"ONNX推理示例已生成: {example_path}")


if __name__ == "__main__":
    main()