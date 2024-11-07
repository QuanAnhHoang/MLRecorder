from dataclasses import dataclass
from typing import Dict, Optional
import yaml
import os

@dataclass
class ModelConfig:
    input_size: tuple[int, int] = (224, 224)  # Standard input size for vision models
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    confidence_threshold: float = 0.7
    max_latency_ms: int = 500

@dataclass
class RecorderConfig:
    capture_interval_ms: int = 100
    max_memory_mb: int = 512
    buffer_size: int = 1000
    save_directory: str = "recordings"

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.model_config = ModelConfig()
        self.recorder_config = RecorderConfig()
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
    
    def _load_from_file(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        if 'model' in config_dict:
            self.model_config = ModelConfig(**config_dict['model'])
        if 'recorder' in config_dict:
            self.recorder_config = RecorderConfig(**config_dict['recorder'])
    
    def to_dict(self) -> Dict:
        return {
            'model': self.model_config.__dict__,
            'recorder': self.recorder_config.__dict__
        }
    
    def save(self, config_path: str) -> None:
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def get_default() -> 'Config':
        return Config()