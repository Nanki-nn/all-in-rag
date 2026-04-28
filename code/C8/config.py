"""
RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/C8/cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "deepseek-chat"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理"""
        self.data_path = str(self._resolve_path(self.data_path))
        self.index_save_path = str(self._resolve_path(self.index_save_path))

    @staticmethod
    def _resolve_path(path_value: str) -> Path:
        """将配置路径解析为相对于当前配置文件的绝对路径。"""
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (BASE_DIR / path).resolve()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
