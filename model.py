import torch
import torch.nn as nn
import torchvision.models as models


class HashingModel(nn.Module):
    def __init__(self, bit):
        super(HashingModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # 定义哈希头部（两层全连接层）
        self.hash_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, bit),
            nn.Tanh()
        )
        # 定义潜在特征头部（两层全连接层）
        self.latent_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # 生成哈希码 b
        b = self.hash_head(x)
        b = torch.sign(b)
        # 生成潜在特征 z 并进行L2规范化
        z = self.latent_head(x)
        z = nn.functional.normalize(z, p=2, dim=1)
        return b, z