import click
import os
import sys
import json
import paddle
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from paddlenlp.data import Stack, Pad, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor
from utils.logger import global_logger as logger
from utils.metrics import MetricsCalculator


@click.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help='训练好的模型路径')
@click.option('--dataset', '-d', type=click.Path(exists=True), required=True, help='测试数据集路径')
@click.option('--batch-size', '-b', type=int, default=32, help='批量大小')
@click.option('--device', '-dev', type=str, default=None, help='指定运行设备')
def main(model: str, dataset: str, batch_size: int, device: str):
    """ERNIE 3.0 Mini 模型评估脚本"""
    # 1. 设置设备
    if device:
        paddle.set_device(device)
    else:
        device = paddle.get_device()
    logger.info(f"使用设备: {device}")
    
    # 2. 初始化数据处理器
    data_processor = DataProcessor()
    
    # 3. 加载标签映射
    label_mapping_path = os.path.join(model, 'label_mapping.json')
    if not os.path.exists(label_mapping_path):
        logger.error(f"未找到标签映射文件: {label_mapping_path}")
        return
    
    label_mapping = data_processor.load_json(label_mapping_path)
    num_classes = len(label_mapping)
    id_to_label = {v: k for k, v in label_mapping.items()}
    logger.info(f"标签映射: {label_mapping}, 类别数: {num_classes}")
    
    # 4. 加载测试数据集
    logger.info(f"加载测试数据集: {dataset}")
    test_data = data_processor.load_jsonl(dataset)
    logger.info(f"测试集大小: {len(test_data)}")
    
    # 5. 转换标签为ID
    test_data = data_processor.convert_labels_to_ids(test_data, label_mapping)
    
    # 6. 加载模型和分词器
    logger.info(f"加载模型: {model}")
    tokenizer = ErnieTokenizer.from_pretrained(model)
    model_obj = ErnieForSequenceClassification.from_pretrained(model, num_classes=num_classes)
    model_obj.eval()
    
    # 7. 数据预处理函数
    def preprocess_function(examples):
        texts = [example['text'] for example in examples]
        labels = [example['label_id'] for example in examples]
        
        # 分词
        tokenized_inputs = tokenizer(
            texts,
            max_seq_len=128,  # 使用默认值，可根据需要调整
            padding=True,
            truncation=True,
            return_token_type_ids=True
        )
        
        return tokenized_inputs['input_ids'], tokenized_inputs['token_type_ids'], labels
    
    # 8. 处理测试数据
    test_processed = []
    for example in test_data:
        input_ids, token_type_ids, label_id = preprocess_function([example])
        test_processed.append([input_ids[0], token_type_ids[0], label_id[0]])
    
    # 9. 创建数据加载器
    def batchify_fn(samples):
        fn = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64")  # labels
        )
        return fn(samples)
    
    test_loader = DataLoader(
        dataset=test_processed,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=batchify_fn,
        num_workers=0
    )
    
    # 9. 进行预测
    logger.info("开始模型评估...")
    
    y_true = []
    y_pred = []
    
    with paddle.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids, token_type_ids, labels = batch
            
            # 前向传播
            logits = model_obj(input_ids, token_type_ids)
            
            # 获取预测结果
            predictions = paddle.argmax(logits, axis=1)
            
            # 保存结果
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.numpy().tolist())
            
            # 打印进度
            if (step + 1) % 10 == 0:
                logger.info(f"处理批次: {step + 1}/{len(test_loader)}")
    
    # 10. 计算评估指标
    logger.info("计算评估指标...")
    metrics_calculator = MetricsCalculator()
    # 显式指定所有类别，确保混淆矩阵形状正确
    all_labels = list(range(num_classes))
    metrics = metrics_calculator.calculate_multiclass_metrics(y_true, y_pred, labels=all_labels)
    
    # 11. 打印评估结果
    logger.info("模型评估结果:")
    metrics_calculator.print_metrics(metrics)
    
    # 12. 保存评估报告
    eval_report_path = os.path.join(model, 'eval_report.json')
    with open(eval_report_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估报告已保存到: {eval_report_path}")
    
    # 13. 生成详细的预测结果
    detailed_results = []
    for i, example in enumerate(test_data):
        detailed_results.append({
            "id": example["id"],
            "text": example["text"],
            "true_label": id_to_label[y_true[i]],
            "predicted_label": id_to_label[y_pred[i]],
            "true_label_id": y_true[i],
            "predicted_label_id": y_pred[i],
            "is_correct": y_true[i] == y_pred[i]
        })
    
    # 保存详细预测结果
    predictions_path = os.path.join(model, 'predictions.jsonl')
    data_processor.save_jsonl(detailed_results, predictions_path)
    logger.info(f"详细预测结果已保存到: {predictions_path}")
    
    # 14. 计算并打印混淆矩阵
    logger.info("混淆矩阵:")
    conf_matrix = metrics['confusion_matrix']
    
    # 打印混淆矩阵表头
    logger.info("\t" + "\t".join(id_to_label[i] for i in range(num_classes)))
    
    # 打印混淆矩阵内容
    for i in range(num_classes):
        row = id_to_label[i] + "\t"
        row += "\t".join(str(conf_matrix[i][j]) for j in range(num_classes))
        logger.info(row)
    
    logger.info("模型评估完成")


if __name__ == "__main__":
    main()