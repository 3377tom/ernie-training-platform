import json
import os
import random
from typing import List, Dict, Tuple, Any
import jsonlines
from sklearn.model_selection import train_test_split


class DataProcessor:
    """数据处理工具类"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                data.append(item)
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        """保存为JSONL文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with jsonlines.open(file_path, 'w') as writer:
            writer.write_all(data)
    
    def load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_json(self, data: Dict, file_path: str):
        """保存为JSON文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def split_dataset(self, data: List[Dict], test_size: float = 0.2, val_size: float = 0.2) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """划分数据集为训练集、验证集和测试集"""
        train_val, test = train_test_split(data, test_size=test_size, random_state=self.seed)
        train, val = train_test_split(train_val, test_size=val_size, random_state=self.seed)
        return train, val, test
    
    def process_raw_data(self, raw_data: List[str], labels: List[str] = None) -> List[Dict]:
        """处理原始数据，转换为标注格式"""
        processed_data = []
        for idx, text in enumerate(raw_data):
            item = {
                "id": f"sample_{idx:06d}",
                "text": text.strip(),
                "label": labels[idx] if labels else "",
                "meta": {
                    "source": "raw",
                    "timestamp": ""
                }
            }
            processed_data.append(item)
        return processed_data
    
    def create_label_mapping(self, data: List[Dict]) -> Dict[str, int]:
        """创建标签映射"""
        labels = set(item["label"] for item in data)
        label_mapping = {label: idx for idx, label in enumerate(sorted(labels))}
        return label_mapping
    
    def convert_labels_to_ids(self, data: List[Dict], label_mapping: Dict[str, int]) -> List[Dict]:
        """将标签转换为ID"""
        for item in data:
            if "label" in item and item["label"] in label_mapping:
                item["label_id"] = label_mapping[item["label"]]
        return data
    
    def shuffle_data(self, data: List[Dict]) -> List[Dict]:
        """打乱数据顺序"""
        shuffled = data.copy()
        random.shuffle(shuffled)
        return shuffled
    
    def filter_data(self, data: List[Dict], condition) -> List[Dict]:
        """根据条件过滤数据"""
        return [item for item in data if condition(item)]
    
    def sample_data(self, data: List[Dict], sample_size: int) -> List[Dict]:
        """随机采样数据"""
        if sample_size >= len(data):
            return data.copy()
        return random.sample(data, sample_size)
    
    def merge_datasets(self, datasets: List[List[Dict]]) -> List[Dict]:
        """合并多个数据集"""
        merged = []
        for dataset in datasets:
            merged.extend(dataset)
        return merged
    
    def validate_data_format(self, data: List[Dict]) -> Tuple[bool, str]:
        """验证数据格式是否正确"""
        required_fields = ["id", "text", "label"]
        
        for idx, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    return False, f"样本 {idx} 缺少必填字段: {field}"
            
            if not isinstance(item["id"], str):
                return False, f"样本 {idx} 的id字段必须是字符串"
            
            if not isinstance(item["text"], str):
                return False, f"样本 {idx} 的text字段必须是字符串"
            
            if not isinstance(item["label"], (str, int)):
                return False, f"样本 {idx} 的label字段必须是字符串或整数"
        
        return True, "数据格式验证通过"
    
    def get_data_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not data:
            return {}
        
        # 计算文本长度统计
        text_lengths = [len(item["text"]) for item in data]
        labels = [item["label"] for item in data]
        unique_labels = set(labels)
        
        return {
            "total_samples": len(data),
            "unique_labels": len(unique_labels),
            "label_distribution": {label: labels.count(label) for label in unique_labels},
            "text_length": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "average": sum(text_lengths) / len(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths) // 2]
            }
        }