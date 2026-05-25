import torch
import torch.nn as nn
import torchvision.models as models


class ADEfficientNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape

        x = x.reshape(B * S, C, H, W)

        features = self.backbone(x)
        features = features.reshape(B, S, -1)

        max_pooled, _ = torch.max(features, dim=1)
        avg_pooled = torch.mean(features, dim=1)

        pooled_features = torch.cat([max_pooled, avg_pooled], dim=1)

        return self.classifier(pooled_features)


class SingleSliceWrapper(nn.Module):
    def __init__(self, full_model):
        super().__init__()

        self.backbone = full_model.backbone
        self.classifier = full_model.classifier

    def forward(self, x):
        # x: (B, C, H, W)

        features = self.backbone(x)

        # Duplicate single-slice features to match CatPool classifier input.
        pooled = torch.cat([features, features], dim=1)

        return self.classifier(pooled)


class CascadeSliceModel(nn.Module):
    def __init__(self, ss_stage1, ss_stage2):
        super().__init__()

        self.ss_stage1 = ss_stage1
        self.ss_stage2 = ss_stage2

    def forward(self, x):
        # x: (B, C, H, W)

        s1_prob = torch.softmax(self.ss_stage1(x), dim=1)
        s2_prob = torch.softmax(self.ss_stage2(x), dim=1)

        p_cn = s1_prob[:, 0:1]
        p_mci = s1_prob[:, 1:2] * s2_prob[:, 0:1]
        p_ad = s1_prob[:, 1:2] * s2_prob[:, 1:2]

        return torch.cat([p_cn, p_mci, p_ad], dim=1)