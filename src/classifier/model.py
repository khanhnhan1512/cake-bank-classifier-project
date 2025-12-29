import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B2_Weights

class LivenessDetectionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        
        # 1. Load Backbone (EfficientNet-B2)
        # Sử dụng weights enum mới thay vì load_url thủ công
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)
        
        # 2. Thay thế Classifier Head
        # EfficientNet B2 có output features là 1408
        num_features = self.backbone.classifier[1].in_features
        
        # Giữ nguyên cấu trúc head phức tạp của bạn để model học tốt hơn
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

if __name__ == "__main__":
    # Test nhanh kích thước model
    import torch
    model = LivenessDetectionModel()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Mong đợi: [2, 2]