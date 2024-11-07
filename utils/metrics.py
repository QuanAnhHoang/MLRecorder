from typing import Dict, List
import time
from dataclasses import dataclass, field
import json
import numpy as np

@dataclass
class PerformanceMetrics:
    latencies: List[float] = field(default_factory=list)
    detection_accuracies: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

class MetricsTracker:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = None
    
    def start_operation(self) -> None:
        self.start_time = time.time()
    
    def end_operation(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Operation not started")
        
        latency = (time.time() - self.start_time) * 1000  # Convert to milliseconds
        self.metrics.latencies.append(latency)
        self.metrics.timestamps.append(time.time())
        return latency
    
    def record_accuracy(self, accuracy: float) -> None:
        self.metrics.detection_accuracies.append(accuracy)
    
    def record_memory(self, memory_mb: float) -> None:
        self.metrics.memory_usage.append(memory_mb)
    
    def get_summary(self) -> Dict:
        if not self.metrics.latencies:
            return {"error": "No metrics recorded"}
        
        return {
            "latency": {
                "mean": np.mean(self.metrics.latencies),
                "median": np.median(self.metrics.latencies),
                "p95": np.percentile(self.metrics.latencies, 95),
                "max": max(self.metrics.latencies)
            },
            "accuracy": {
                "mean": np.mean(self.metrics.detection_accuracies) if self.metrics.detection_accuracies else None
            },
            "memory": {
                "mean": np.mean(self.metrics.memory_usage) if self.metrics.memory_usage else None,
                "max": max(self.metrics.memory_usage) if self.metrics.memory_usage else None
            }
        }
    
    def save_metrics(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def clear(self) -> None:
        self.metrics = PerformanceMetrics()
        self.start_time = None