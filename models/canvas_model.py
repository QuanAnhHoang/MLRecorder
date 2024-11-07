import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from .shared_features import BaseElementDetector

class CanvasFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Deep feature extraction
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                    blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 
                                  kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

class CanvasElementDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = CanvasFeatureExtractor()
        
        # Region Proposal Network
        self.rpn = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv2d(256, 2, kernel_size=1)  # Object vs background
        self.rpn_bbox = nn.Conv2d(256, 4, kernel_size=1)  # Bounding box regression
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # 4 interaction types
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        
        # RPN
        rpn_features = self.rpn(features)
        objectness = self.rpn_cls(rpn_features)
        bbox_deltas = self.rpn_bbox(rpn_features)
        
        # Classification
        class_scores = self.classifier(features)
        
        return bbox_deltas, objectness, class_scores
    
    def predict(self, x: torch.Tensor, confidence_threshold: float = 0.7) -> Dict[str, Optional[torch.Tensor]]:
        bbox_deltas, objectness, class_scores = self.forward(x)
        
        # Process objectness scores
        obj_probs = torch.softmax(objectness, dim=1)
        max_obj_prob, _ = torch.max(obj_probs[:, 1], dim=0)  # Probability of being an object
        
        # Process classification scores
        class_probs = torch.softmax(class_scores, dim=1)
        max_class_prob, predicted_class = torch.max(class_probs, dim=1)
        
        if max_obj_prob.item() < confidence_threshold or max_class_prob.item() < confidence_threshold:
            return {
                'bbox': None,
                'class': None,
                'confidence': min(max_obj_prob.item(), max_class_prob.item())
            }
        
        return {
            'bbox': bbox_deltas,
            'class': predicted_class,
            'confidence': min(max_obj_prob.item(), max_class_prob.item())
        }

class CanvasModelTrainer:
    def __init__(self, model: CanvasElementDetector,
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.bbox_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.obj_criterion = nn.CrossEntropyLoss()
        
    def train_step(self, image: torch.Tensor, target_bbox: torch.Tensor,
                  target_obj: torch.Tensor, target_class: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move inputs to device
        image = image.to(self.device)
        target_bbox = target_bbox.to(self.device)
        target_obj = target_obj.to(self.device)
        target_class = target_class.to(self.device)
        
        # Forward pass
        bbox_deltas, objectness, class_scores = self.model(image)
        
        # Calculate losses
        bbox_loss = self.bbox_criterion(bbox_deltas, target_bbox)
        obj_loss = self.obj_criterion(objectness.view(-1, 2), target_obj.view(-1))
        cls_loss = self.cls_criterion(class_scores, target_class)
        
        total_loss = bbox_loss + obj_loss + cls_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'bbox_loss': bbox_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_bbox_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image, target_bbox, target_obj, target_class in val_loader:
                image = image.to(self.device)
                target_bbox = target_bbox.to(self.device)
                target_obj = target_obj.to(self.device)
                target_class = target_class.to(self.device)
                
                bbox_deltas, objectness, class_scores = self.model(image)
                
                total_bbox_loss += self.bbox_criterion(bbox_deltas, target_bbox).item()
                total_obj_loss += self.obj_criterion(objectness.view(-1, 2), 
                                                   target_obj.view(-1)).item()
                total_cls_loss += self.cls_criterion(class_scores, target_class).item()
                
                _, predicted = torch.max(class_scores, 1)
                total += target_class.size(0)
                correct += (predicted == target_class).sum().item()
        
        return {
            'val_bbox_loss': total_bbox_loss / len(val_loader),
            'val_obj_loss': total_obj_loss / len(val_loader),
            'val_cls_loss': total_cls_loss / len(val_loader),
            'val_accuracy': correct / total
        }