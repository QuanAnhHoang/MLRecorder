import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from PIL import Image
import base64
import io
import cv2
from ..models.dom_model import DOMElementDetector
from ..models.canvas_model import CanvasElementDetector
from ..utils.metrics import MetricsTracker

class ElementDetectorManager:
    def __init__(self, confidence_threshold: float = 0.7):
        self.dom_model = DOMElementDetector()
        self.canvas_model = CanvasElementDetector()
        self.confidence_threshold = confidence_threshold
        self.metrics = MetricsTracker()
        
        # Load models if available
        self._load_models()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dom_model.to(self.device)
        self.canvas_model.to(self.device)
        
        # Set models to evaluation mode
        self.dom_model.eval()
        self.canvas_model.eval()
        
    def _load_models(self):
        try:
            self.dom_model.load_state_dict(
                torch.load('data/models/dom_model.pth', map_location='cpu')
            )
            self.canvas_model.load_state_dict(
                torch.load('data/models/canvas_model.pth', map_location='cpu')
            )
        except FileNotFoundError:
            print("Warning: Model weights not found. Using initialized models.")
    
    def _preprocess_image(self, image_data: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image_data, str):  # Base64 string
            try:
                # Decode base64 string
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image = Image.open(io.BytesIO(image_bytes))
            except:
                raise ValueError("Invalid base64 image string")
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Unsupported image format")
        
        # Resize and normalize
        image = image.resize((224, 224))
        image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def _preprocess_dom(self, dom_string: str) -> torch.Tensor:
        # Simple tokenization (in practice, you'd want a more sophisticated tokenizer)
        tokens = dom_string.split()
        # Convert to indices (simplified for demonstration)
        token_ids = [hash(token) % 10000 for token in tokens]
        # Pad or truncate to fixed length
        token_ids = token_ids[:512] + [0] * max(0, 512 - len(token_ids))
        return torch.tensor(token_ids).unsqueeze(0)
    
    def detect_elements_dom(self, image_data: Union[str, np.ndarray, Image.Image], 
                          dom_string: str) -> Dict[str, Union[Dict, float]]:
        self.metrics.start_operation()
        
        try:
            image_tensor = self._preprocess_image(image_data).to(self.device)
            dom_tensor = self._preprocess_dom(dom_string).to(self.device)
            
            with torch.no_grad():
                result = self.dom_model.predict(
                    image_tensor,
                    dom_tensor,
                    confidence_threshold=self.confidence_threshold
                )
            
            latency = self.metrics.end_operation()
            
            if result['bbox'] is not None:
                result['bbox'] = result['bbox'].cpu().numpy().tolist()
            if result['class'] is not None:
                result['class'] = result['class'].cpu().numpy().tolist()
            
            return {
                'detection': result,
                'latency_ms': latency
            }
            
        except Exception as e:
            self.metrics.end_operation()
            return {
                'error': str(e),
                'detection': None,
                'latency_ms': -1
            }
    
    def detect_elements_canvas(self, image_data: Union[str, np.ndarray, Image.Image]) -> Dict[str, Union[Dict, float]]:
        self.metrics.start_operation()
        
        try:
            image_tensor = self._preprocess_image(image_data).to(self.device)
            
            with torch.no_grad():
                result = self.canvas_model.predict(
                    image_tensor,
                    confidence_threshold=self.confidence_threshold
                )
            
            latency = self.metrics.end_operation()
            
            if result['bbox'] is not None:
                result['bbox'] = result['bbox'].cpu().numpy().tolist()
            if result['class'] is not None:
                result['class'] = result['class'].cpu().numpy().tolist()
            
            return {
                'detection': result,
                'latency_ms': latency
            }
            
        except Exception as e:
            self.metrics.end_operation()
            return {
                'error': str(e),
                'detection': None,
                'latency_ms': -1
            }
    
    def get_performance_metrics(self) -> Dict:
        return self.metrics.get_summary()
    
    def clear_metrics(self) -> None:
        self.metrics.clear()
    
    def get_element_coordinates(self, bbox: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert normalized bbox coordinates to pixel coordinates"""
        x1, y1, x2, y2 = bbox
        return (
            int(x1 * 224),  # Normalized to input size
            int(y1 * 224),
            int(x2 * 224),
            int(y2 * 224)
        )
    
    def get_interaction_type(self, class_id: int) -> str:
        interaction_types = ['click', 'input', 'hover', 'scroll']
        return interaction_types[class_id]