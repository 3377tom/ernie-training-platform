from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
from typing import Dict, List, Any


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, average: str = 'macro'):
        self.average = average
    
    def calculate_classification_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """计算分类任务的评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average),
            'recall': recall_score(y_true, y_pred, average=self.average),
            'f1_score': f1_score(y_true, y_pred, average=self.average)
        }
        return metrics
    
    def calculate_multiclass_metrics(self, y_true: List[int], y_pred: List[int], labels: List[int] = None) -> Dict[str, Any]:
        """计算多分类任务的详细评估指标"""
        # 基础指标
        metrics = self.calculate_classification_metrics(y_true, y_pred)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        
        # 分类报告（包含每个类别的指标）
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return metrics
    
    def calculate_regression_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """计算回归任务的评估指标"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        return metrics
    
    def calculate_sequence_labeling_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
        """计算序列标注任务的评估指标"""
        # 扁平化标签列表
        flat_true = [label for seq in y_true for label in seq]
        flat_pred = [label for seq in y_pred for label in seq]
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(flat_true, flat_pred),
            'precision': precision_score(flat_true, flat_pred, average=self.average, zero_division=0),
            'recall': recall_score(flat_true, flat_pred, average=self.average, zero_division=0),
            'f1_score': f1_score(flat_true, flat_pred, average=self.average, zero_division=0)
        }
        
        return metrics
    
    def calculate_ner_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
        """计算命名实体识别任务的评估指标"""
        # 使用IOB或BIOES格式计算实体级别的指标
        true_entities = self._get_entities(y_true)
        pred_entities = self._get_entities(y_pred)
        
        # 计算精确匹配
        exact_match = 0
        for true_ent in true_entities:
            if true_ent in pred_entities:
                exact_match += 1
        
        # 计算部分匹配（使用集合交集）
        # 注意：这是一个简化的实现，实际NER评估通常更复杂
        precision = exact_match / len(pred_entities) if pred_entities else 0
        recall = exact_match / len(true_entities) if true_entities else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'exact_match': exact_match,
            'true_entities': len(true_entities),
            'pred_entities': len(pred_entities),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # 添加标签级别的指标
        flat_true = [label for seq in y_true for label in seq]
        flat_pred = [label for seq in y_pred for label in seq]
        metrics['token_level'] = {
            'accuracy': accuracy_score(flat_true, flat_pred),
            'precision': precision_score(flat_true, flat_pred, average=self.average, zero_division=0),
            'recall': recall_score(flat_true, flat_pred, average=self.average, zero_division=0),
            'f1_score': f1_score(flat_true, flat_pred, average=self.average, zero_division=0)
        }
        
        return metrics
    
    def _get_entities(self, sequences: List[List[str]]) -> List[tuple]:
        """从序列中提取实体，支持IOB和BIOES格式"""
        entities = []
        for seq_idx, seq in enumerate(sequences):
            i = 0
            while i < len(seq):
                label = seq[i]
                if label.startswith('B-') or label.startswith('I-') or label.startswith('E-') or label.startswith('S-'):
                    entity_type = label.split('-')[1]
                    start = i
                    
                    # 处理BIOES格式
                    if label.startswith('S-'):
                        # 单个实体
                        entities.append((seq_idx, start, start + 1, entity_type))
                        i += 1
                    elif label.startswith('B-'):
                        # 开始一个实体
                        j = i + 1
                        while j < len(seq) and (seq[j].startswith('I-') or seq[j].startswith('E-')):
                            if seq[j].startswith('E-') and seq[j].split('-')[1] == entity_type:
                                # 实体结束
                                entities.append((seq_idx, start, j + 1, entity_type))
                                i = j + 1
                                break
                            elif seq[j].startswith('I-') and seq[j].split('-')[1] == entity_type:
                                j += 1
                            else:
                                break
                        else:
                            # 如果没有E-标签，使用I-标签直到结束
                            if j > i:
                                entities.append((seq_idx, start, j, entity_type))
                            i = j
                    else:
                        i += 1
                else:
                    i += 1
        
        return entities
    
    def print_metrics(self, metrics: Dict[str, Any], title: str = "Evaluation Metrics"):
        """打印评估指标"""
        print(f"\n{title}")
        print("=" * 50)
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        print(f"  {sub_key}: {sub_value:.4f}")
                    else:
                        print(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print("=" * 50)