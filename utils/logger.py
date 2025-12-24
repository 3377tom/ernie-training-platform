from loguru import logger
import os
import sys


def setup_logger(log_dir=None, log_name="training.log"):
    """设置日志配置"""
    # 移除默认的日志配置
    logger.remove()
    
    # 基本日志格式
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level="INFO",
        enqueue=True
    )
    
    # 添加文件输出（如果提供了log_dir）
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_name)
        
        logger.add(
            log_file,
            format=log_format,
            level="DEBUG",
            enqueue=True,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    return logger


# 创建全局logger实例
global_logger = setup_logger()