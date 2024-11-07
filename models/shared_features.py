from typing import Tuple, Optional
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return self.dropout(x)

class ElementLocalization(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, 4, kernel_size=1)  # 4 for bounding box coordinates
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)

class InteractionClassifier(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(features)
        x = torch.flatten(x, 1)
        return self.fc(x)

class BaseElementDetector(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 4):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_channels)
        self.localizer = ElementLocalization()
        self.classifier = InteractionClassifier(num_classes=num_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        bbox = self.localizer(features)
        class_scores = self.classifier(features)
        return bbox, class_scores
    
    def predict(self, x: torch.Tensor, confidence_threshold: float = 0.7) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bbox, class_scores = self.forward(x)
        probabilities = torch.softmax(class_scores, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        
        if max_prob.item() < confidence_threshold:
            return None, None
            
        return bbox, predicted_class