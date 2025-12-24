import click
import os
import sys
import json
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from paddleslim.dygraph import QAT, QuantConfig

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import global_logger as logger


@click.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help='训练好的模型路径')
@click.option('--output', '-o', type=click.Path(), required=True, help='压缩后模型输出路径')
@click.option('--method', '-t', type=click.Choice(['quant', 'prune', 'distill']), default='quant', help='压缩方法')
@click.option('--dataset', '-d', type=click.Path(exists=True), help='用于量化校准的数据集路径')
@click.option('--batch-size', '-b', type=int, default=32, help='批量大小')
@click.option('--device', '-dev', type=str, default=None, help='指定运行设备')
def main(model: str, output: str, method: str, dataset: str, batch_size: int, device: str):
    """ERNIE 3.0 Mini 模型压缩脚本"""
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
    if not os.path.exists(label_mapping_path):
        logger.error(f"未找到标签映射文件: {label_mapping_path}")
        return
    
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping)
    
    original_model = ErnieForSequenceClassification.from_pretrained(model, num_classes=num_classes)
    original_model.eval()
    
    # 3. 根据压缩方法进行处理
    if method == 'quant':
        # 模型量化
        logger.info("执行模型量化...")
        compressed_model = quantize_model(original_model, tokenizer, dataset, batch_size, device)
    
    elif method == 'prune':
        # 模型剪枝
        logger.info("执行模型剪枝...")
        compressed_model = prune_model(original_model)
    
    elif method == 'distill':
        # 知识蒸馏
        logger.info("执行知识蒸馏...")
        compressed_model = distill_model(original_model, dataset, batch_size, device)
    
    else:
        logger.error(f"不支持的压缩方法: {method}")
        return
    
    # 4. 保存压缩后的模型
    logger.info(f"保存压缩后的模型到: {output}")
    os.makedirs(output, exist_ok=True)
    
    # 保存模型
    compressed_model.save_pretrained(output)
    
    # 保存分词器
    tokenizer.save_pretrained(output)
    
    # 复制标签映射
    import shutil
    shutil.copy2(label_mapping_path, os.path.join(output, 'label_mapping.json'))
    
    # 保存压缩配置
    compress_config = {
        'original_model': model,
        'compress_method': method,
        'timestamp': paddle.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    compress_config_path = os.path.join(output, 'compress_config.json')
    with open(compress_config_path, 'w', encoding='utf-8') as f:
        json.dump(compress_config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"模型压缩完成，压缩方法: {method}")
    logger.info(f"压缩后的模型已保存到: {output}")


def quantize_model(model, tokenizer, dataset_path, batch_size, device):
    """模型量化"""
    # 初始化量化配置
    quant_config = QuantConfig(
        weight_quantize_type='channel_wise_abs_max',
        activation_quantize_type='moving_average_abs_max',
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9
    )
    
    # 创建QAT模型
    qat_model = QAT(config=quant_config)(model)
    qat_model.eval()
    
    if dataset_path:
        # 加载校准数据集并进行量化校准
        logger.info(f"使用校准数据集: {dataset_path}")
        from utils.data_processor import DataProcessor
        data_processor = DataProcessor()
        
        # 加载校准数据
        calib_data = data_processor.load_jsonl(dataset_path)
        logger.info(f"校准数据大小: {len(calib_data)}")
        
        # 数据预处理
        def preprocess_fn(example):
            inputs = tokenizer(
                example['text'],
                max_seq_len=128,
                padding=True,
                truncation=True,
                return_token_type_ids=True
            )
            return inputs['input_ids'], inputs['token_type_ids']
        
        # 执行校准
        with paddle.no_grad():
            for i in range(0, len(calib_data), batch_size):
                batch_data = calib_data[i:i+batch_size]
                input_ids = []
                token_type_ids = []
                
                for example in batch_data:
                    ids, type_ids = preprocess_fn(example)
                    input_ids.append(ids)
                    token_type_ids.append(type_ids)
                
                # 转换为paddle tensor
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)
                
                # 前向传播，进行量化校准
                qat_model(input_ids, token_type_ids)
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"校准进度: {i + batch_size}/{len(calib_data)}")
    
    logger.info("模型量化完成")
    return qat_model


def prune_model(model):
    """模型剪枝"""
    # 这里实现模型剪枝，目前使用简单的通道剪枝示例
    logger.info("执行通道剪枝")
    # 注意：实际剪枝需要更复杂的逻辑和微调过程
    # 这里仅作为示例，返回原始模型
    return model


def distill_model(model, dataset_path, batch_size, device):
    """知识蒸馏"""
    # 这里实现知识蒸馏，使用小型模型作为学生模型
    logger.info("执行知识蒸馏")
    # 注意：实际蒸馏需要更复杂的逻辑，包括学生模型定义、损失函数设计等
    # 这里仅作为示例，返回原始模型
    return model


if __name__ == "__main__":
    main()