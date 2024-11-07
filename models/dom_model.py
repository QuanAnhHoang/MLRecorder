import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from .shared_features import BaseElementDetector

class DOMFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)  # Vocabulary size of 10000
        self.lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return output

class DOMElementDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_detector = BaseElementDetector(input_channels=3, num_classes=4)
        self.dom_feature_extractor = DOMFeatureExtractor()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # 4 classes: click, input, hover, scroll
        )
        
    def forward(self, image: torch.Tensor, dom_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process visual features
        bbox, visual_features = self.visual_detector(image)
        
        # Process DOM features
        dom_features = self.dom_feature_extractor(dom_sequence)
        dom_features = torch.mean(dom_features, dim=1)  # Average pooling over sequence
        
        # Combine features
        combined = torch.cat([visual_features.view(visual_features.size(0), -1), 
                            dom_features], dim=1)
        fused = self.fusion(combined)
        
        # Final classification
        class_scores = self.classifier(fused)
        
        return bbox, class_scores

    def predict(self, image: torch.Tensor, dom_sequence: torch.Tensor, 
                confidence_threshold: float = 0.7) -> Dict[str, Optional[torch.Tensor]]:
        bbox, class_scores = self.forward(image, dom_sequence)
        probabilities = torch.softmax(class_scores, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        
        if max_prob.item() < confidence_threshold:
            return {
                'bbox': None,
                'class': None,
                'confidence': max_prob.item()
            }
            
        return {
            'bbox': bbox,
            'class': predicted_class,
            'confidence': max_prob.item()
        }
    
    def get_dom_embedding(self, dom_sequence: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.dom_feature_extractor(dom_sequence)

class DOMModelTrainer:
    def __init__(self, model: DOMElementDetector, 
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.bbox_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()
        
    def train_step(self, image: torch.Tensor, dom_sequence: torch.Tensor, 
                  target_bbox: torch.Tensor, target_class: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move inputs to device
        image = image.to(self.device)
        dom_sequence = dom_sequence.to(self.device)
        target_bbox = target_bbox.to(self.device)
        target_class = target_class.to(self.device)
        
        # Forward pass
        bbox, class_scores = self.model(image, dom_sequence)
        
        # Calculate losses
        bbox_loss = self.bbox_criterion(bbox, target_bbox)
        cls_loss = self.cls_criterion(class_scores, target_class)
        total_loss = bbox_loss + cls_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'bbox_loss': bbox_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_bbox_loss = 0.0
        total_cls_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image, dom_sequence, target_bbox, target_class in val_loader:
                image = image.to(self.device)
                dom_sequence = dom_sequence.to(self.device)
                target_bbox = target_bbox.to(self.device)
                target_class = target_class.to(self.device)
                
                bbox, class_scores = self.model(image, dom_sequence)
                
                total_bbox_loss += self.bbox_criterion(bbox, target_bbox).item()
                total_cls_loss += self.cls_criterion(class_scores, target_class).item()
                
                _, predicted = torch.max(class_scores, 1)
                total += target_class.size(0)
                correct += (predicted == target_class).sum().item()
        
        return {
            'val_bbox_loss': total_bbox_loss / len(val_loader),
            'val_cls_loss': total_cls_loss / len(val_loader),
            'val_accuracy': correct / total
        }