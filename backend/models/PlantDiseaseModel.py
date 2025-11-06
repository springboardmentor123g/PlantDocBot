# models/PlantDiseaseModel.py

import torch.nn as nn

# This class is now an EXACT copy of the SimpleCNN from your training code.
# I've only renamed it to PlantDiseaseModel to match your main.py imports.
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # This adaptive pooling layer is important
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return x