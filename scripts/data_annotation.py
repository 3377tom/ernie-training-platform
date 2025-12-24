import click
import os
import sys
import glob
from typing import List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor
from utils.logger import global_logger as logger


@click.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='原始数据输入目录')
@click.option('--output', '-o', type=click.Path(), required=True, help='标注数据输出目录')
@click.option('--split', '-s', is_flag=True, default=False, help='是否自动划分数据集')
@click.option('--test-size', type=float, default=0.2, help='测试集比例')
@click.option('--val-size', type=float, default=0.2, help='验证集比例')
def main(input: str, output: str, split: bool, test_size: float, val_size: float):
    """数据标注脚本：将原始数据转换为标注格式并进行预处理"""
    logger.info(f"开始数据标注流程，输入目录: {input}, 输出目录: {output}")
    
    # 初始化数据处理器
    data_processor = DataProcessor()
    
    # 1. 读取原始数据
    raw_data = []
    labels = []
    
    # 支持多种文件格式
    input_files = []
    for ext in ['*.txt', '*.csv', '*.json', '*.jsonl']:
        input_files.extend(glob.glob(os.path.join(input, ext)))
    
    if not input_files:
        logger.error(f"在输入目录 {input} 中未找到任何数据文件")
        return
    
    logger.info(f"找到 {len(input_files)} 个数据文件")
    
    # 根据文件格式读取数据
    for file_path in input_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                raw_data.extend(lines)
        
        elif file_ext == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) >= 1:
                        raw_data.append(row[0])
                        if len(row) >= 2:
                            labels.append(row[1])
        
        elif file_ext == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            raw_data.append(item['text'])
                            if 'label' in item:
                                labels.append(item['label'])
                elif isinstance(data, dict) and 'data' in data:
                    for item in data['data']:
                        if 'text' in item:
                            raw_data.append(item['text'])
                            if 'label' in item:
                                labels.append(item['label'])
        
        elif file_ext == '.jsonl':
            import jsonlines
            with jsonlines.open(file_path, 'r') as reader:
                for item in reader:
                    if 'text' in item:
                        raw_data.append(item['text'])
                        if 'label' in item:
                            labels.append(item['label'])
    
    logger.info(f"共读取 {len(raw_data)} 条原始数据")
    
    # 2. 处理数据，转换为标注格式
    if labels and len(labels) == len(raw_data):
        processed_data = data_processor.process_raw_data(raw_data, labels)
    else:
        processed_data = data_processor.process_raw_data(raw_data)
    
    logger.info(f"数据处理完成，生成 {len(processed_data)} 条标注数据")
    
    # 3. 验证数据格式
    is_valid, message = data_processor.validate_data_format(processed_data)
    if not is_valid:
        logger.error(f"数据格式验证失败: {message}")
        return
    
    logger.info("数据格式验证通过")
    
    # 4. 保存标注数据
    annotated_output = os.path.join(output, "annotated.jsonl")
    data_processor.save_jsonl(processed_data, annotated_output)
    logger.info(f"标注数据已保存到: {annotated_output}")
    
    # 5. 如果需要，划分数据集
    if split:
        logger.info(f"开始划分数据集，测试集比例: {test_size}, 验证集比例: {val_size}")
        train_data, val_data, test_data = data_processor.split_dataset(processed_data, test_size, val_size)
        
        # 保存划分后的数据集
        train_output = os.path.join(output, "train.jsonl")
        val_output = os.path.join(output, "val.jsonl")
        test_output = os.path.join(output, "test.jsonl")
        
        data_processor.save_jsonl(train_data, train_output)
        data_processor.save_jsonl(val_data, val_output)
        data_processor.save_jsonl(test_data, test_output)
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(train_data)} 条，保存到: {train_output}")
        logger.info(f"  验证集: {len(val_data)} 条，保存到: {val_output}")
        logger.info(f"  测试集: {len(test_data)} 条，保存到: {test_output}")
    
    # 6. 生成数据统计信息
    stats = data_processor.get_data_statistics(processed_data)
    stats_output = os.path.join(output, "data_statistics.json")
    data_processor.save_json(stats, stats_output)
    
    logger.info(f"数据统计信息已保存到: {stats_output}")
    logger.info(f"数据标注流程已完成")


if __name__ == "__main__":
    main()