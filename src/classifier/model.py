import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B2_Weights

class LivenessDetectionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Load Backbone (EfficientNet-B2)
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        
        # Maintain the complex head structure for better learning capability
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 768),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            
            nn.Dropout(p=dropout + 0.05),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(384),
            
            nn.Dropout(p=dropout),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)