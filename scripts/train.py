import click
import os
import sys
import json
import paddle
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.metrics import AccuracyAndF1

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor
from utils.logger import global_logger as logger
from utils.metrics import MetricsCalculator


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='训练配置文件路径')
@click.option('--device', '-d', type=str, default=None, help='指定运行设备，如cpu、gpu:0等')
def main(config: str, device: str):
    """ERNIE 3.0 Mini 模型训练脚本"""
    # 1. 加载配置文件
    logger.info(f"加载训练配置: {config}")
    with open(config, 'r', encoding='utf-8') as f:
        training_config = json.load(f)
    
    # 设置设备
    if device:
        paddle.set_device(device)
    else:
        device = paddle.get_device()
    logger.info(f"使用设备: {device}")
    
    # 2. 初始化数据处理器
    data_processor = DataProcessor(seed=training_config.get('seed', 42))
    
    # 3. 加载数据集
    logger.info("加载训练数据集...")
    train_data = data_processor.load_jsonl(training_config['train_data'])
    val_data = data_processor.load_jsonl(training_config['val_data'])
    
    logger.info(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
    
    # 4. 创建标签映射
    label_mapping = data_processor.create_label_mapping(train_data + val_data)
    num_classes = len(label_mapping)
    logger.info(f"标签映射: {label_mapping}, 类别数: {num_classes}")
    
    # 保存标签映射
    label_mapping_path = os.path.join(training_config['output_dir'], 'label_mapping.json')
    data_processor.save_json(label_mapping, label_mapping_path)
    logger.info(f"标签映射已保存到: {label_mapping_path}")
    
    # 5. 转换标签为ID
    train_data = data_processor.convert_labels_to_ids(train_data, label_mapping)
    val_data = data_processor.convert_labels_to_ids(val_data, label_mapping)
    
    # 6. 加载预训练模型和分词器
    logger.info(f"加载预训练模型: {training_config['model_name']}")
    
    # 初始化分词器
    tokenizer = ErnieTokenizer.from_pretrained(training_config['model_name'])
    
    # 初始化模型
    if training_config.get('pretrained_model_path') and os.path.exists(training_config['pretrained_model_path']):
        # 加载本地预训练模型
        model = ErnieForSequenceClassification.from_pretrained(
            training_config['pretrained_model_path'],
            num_classes=num_classes
        )
        logger.info(f"加载本地预训练模型: {training_config['pretrained_model_path']}")
    else:
        # 从Hugging Face或PaddleNLP模型库加载
        model = ErnieForSequenceClassification.from_pretrained(
            training_config['model_name'],
            num_classes=num_classes
        )
        logger.info(f"从模型库加载预训练模型: {training_config['model_name']}")
    
    # 7. 数据预处理函数
    def preprocess_function(examples):
        texts = [example['text'] for example in examples]
        labels = [example['label_id'] for example in examples]
        
        # 分词
        tokenized_inputs = tokenizer(
            texts,
            max_seq_len=training_config['max_seq_length'],
            padding=True,
            truncation=True,
            return_token_type_ids=True
        )
        
        return tokenized_inputs['input_ids'], tokenized_inputs['token_type_ids'], labels
    
    # 8. 创建数据加载器
    def batchify_fn(samples):
        fn = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64")  # labels
        )
        return fn(samples)
    
    # 处理训练数据
    train_processed = []
    for example in train_data:
        input_ids, token_type_ids, label_id = preprocess_function([example])
        train_processed.append([input_ids[0], token_type_ids[0], label_id[0]])
    
    # 处理验证数据
    val_processed = []
    for example in val_data:
        input_ids, token_type_ids, label_id = preprocess_function([example])
        val_processed.append([input_ids[0], token_type_ids[0], label_id[0]])
    
    # 训练集数据加载器
    train_loader = DataLoader(
        dataset=train_processed,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=batchify_fn,
        num_workers=0
    )
    
    # 验证集数据加载器
    val_loader = DataLoader(
        dataset=val_processed,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=batchify_fn,
        num_workers=0
    )
    
    # 9. 设置优化器和损失函数
    logger.info("初始化优化器和损失函数...")
    
    # 优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=training_config['learning_rate'],
        parameters=model.parameters(),
        weight_decay=training_config['weight_decay']
    )
    
    # 损失函数
    loss_fn = paddle.nn.CrossEntropyLoss()
    
    # 10. 设置评估指标
    metrics_calculator = MetricsCalculator()
    
    # 11. 开始训练
    logger.info("开始模型训练...")
    
    best_accuracy = 0.0
    early_stopping_counter = 0
    
    for epoch in range(training_config['epochs']):
        logger.info(f"Epoch {epoch + 1}/{training_config['epochs']}")
        
        # 训练模式
        model.train()
        total_loss = 0.0
        train_preds = []
        train_labels = []
        
        for step, batch in enumerate(train_loader):
            input_ids, token_type_ids, labels = batch
            
            # 前向传播
            logits = model(input_ids, token_type_ids)
            loss = loss_fn(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if training_config.get('max_grad_norm'):
                paddle.nn.utils.clip_grad_norm_(model.parameters(), training_config['max_grad_norm'])
            
            # 更新参数
            optimizer.step()
            optimizer.clear_grad()
            
            # 保存预测结果和真实标签
            preds = paddle.argmax(logits, axis=1)
            train_preds.extend(preds.numpy().tolist())
            train_labels.extend(labels.numpy().tolist())
            
            total_loss += loss.item()
            
            # 打印日志
            if (step + 1) % training_config['logging_steps'] == 0:
                avg_loss = total_loss / (step + 1)
                # 计算当前指标
                current_preds = train_preds[-len(labels):]
                current_labels = train_labels[-len(labels):]
                current_acc = sum([1 for p, l in zip(current_preds, current_labels) if p == l]) / len(current_preds)
                logger.info(f"Step {step + 1}, Loss: {avg_loss:.4f}, Accuracy: {current_acc:.4f}")
        
        # 训练集最终指标
        train_acc = sum([1 for p, l in zip(train_preds, train_labels) if p == l]) / len(train_preds)
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Train Epoch {epoch + 1} | Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with paddle.no_grad():
            for step, batch in enumerate(val_loader):
                input_ids, token_type_ids, labels = batch
                
                # 前向传播
                logits = model(input_ids, token_type_ids)
                loss = loss_fn(logits, labels)
                
                # 保存预测结果和真实标签
                preds = paddle.argmax(logits, axis=1)
                val_preds.extend(preds.numpy().tolist())
                val_labels.extend(labels.numpy().tolist())
                
                val_loss += loss.item()
        
        # 验证集最终指标
        val_acc = sum([1 for p, l in zip(val_preds, val_labels) if p == l]) / len(val_preds)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Val Epoch {epoch + 1} | Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            early_stopping_counter = 0
            
            # 保存模型
            logger.info(f"保存最佳模型，当前Accuracy: {best_accuracy:.4f}")
            model.save_pretrained(training_config['output_dir'])
            tokenizer.save_pretrained(training_config['output_dir'])
            
            # 保存配置文件
            config_save_path = os.path.join(training_config['output_dir'], 'training_config.json')
            with open(config_save_path, 'w', encoding='utf-8') as f:
                json.dump(training_config, f, ensure_ascii=False, indent=2)
        else:
            early_stopping_counter += 1
            logger.info(f"早停计数: {early_stopping_counter}/{training_config['early_stopping_patience']}")
            
            # 早停检查
            if early_stopping_counter >= training_config['early_stopping_patience']:
                logger.info("触发早停条件，停止训练")
                break
    
    logger.info(f"训练完成，最佳Accuracy: {best_accuracy:.4f}")
    logger.info(f"模型已保存到: {training_config['output_dir']}")


if __name__ == "__main__":
    main()