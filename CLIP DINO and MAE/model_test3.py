import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 定义dataset 类
class MultiView_Dataset(Dataset):
    def __init__(self, features_dict):
        self.clip_features = features_dict["clip_features"]
        self.dino_features = features_dict["dino_features"]
        self.mae_features = features_dict["mae_features"]
        self.labels = features_dict["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.clip_features[idx], self.dino_features[idx], self.mae_features[idx], self.labels[idx]

# 定义模型架构
class MultiView_Fusion_MLP(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768, mae_dim=768, num_classes=196):
        super(MultiView_Fusion_MLP, self).__init__()
        concat_dim = clip_dim + dino_dim + mae_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clip_feat, dino_feat, mae_feat):
        fused_feature = torch.cat([clip_feat, dino_feat, mae_feat], dim=1)
        return self.mlp(fused_feature)

# 评估流程
def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    # 加载测试集特征
    print("正在加载测试集特征")
    features_dict = torch.load("multiview3_test_features.pt", map_location="cpu")
    test_dataset = MultiView_Dataset(features_dict)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 初始化模型并加载权重
    print("正在加载训练好的模型权重")
    model = MultiView_Fusion_MLP().to(device)
    model.load_state_dict(torch.load("multiview3_fusion_mlp.pth", map_location=device))
    
    # 进行模型评估
    model.eval()
    
    correct_top1 = 0
    total = 0

    print("开始在官方测试集上进行评估")
    with torch.no_grad():
        for clip_feat, dino_feat, mae_feat, labels in test_loader:
            clip_feat = clip_feat.to(device).float()
            dino_feat = dino_feat.to(device).float()
            mae_feat = mae_feat.to(device).float()
            labels = labels.to(device).long()
            
            # 模型推理
            logits = model(clip_feat, dino_feat, mae_feat)
            
            # 拿到预测结果并对比真实标签
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

    accuracy = 100 * correct_top1 / total
    
    print("-" * 40)
    print(f"多视图融合模型 (CLIP + DINO + mae) 最终成绩")
    print(f"测试集图片总数: {total} 张")
    print(f"Top-1 准确率: {accuracy:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_model()