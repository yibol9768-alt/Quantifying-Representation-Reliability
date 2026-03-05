import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time


#  构建专门的多视图 Dataset

class MultiView_Dataset(Dataset):
    def __init__(self, features_dict):
        # 只保留图像特征，没有文本特征
        self.clip_features = features_dict["clip_features"]  # [N, 512]
        self.dino_features = features_dict["dino_features"]  # [N, 768]
        self.labels = features_dict["labels"]                # [N] (0~195整数)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.clip_features[idx], self.dino_features[idx], self.labels[idx]


#  构建多视图融合 MLP 模型
class MultiView_Fusion_MLP(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768, num_classes=196):
        super(MultiView_Fusion_MLP, self).__init__()
        
        # 拼接两种视图的维度 512 + 768 = 1280
        concat_dim = clip_dim + dino_dim
        
        # 构建分类器
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.BatchNorm1d(1024), # 加入 BN 层加速收敛
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clip_feat, dino_feat):
        # 沿着特征维度进行拼接: fused_feature [Batch, 1280]
        fused_feature = torch.cat([clip_feat, dino_feat], dim=1)
        # 输出分类 logits
        logits = self.mlp(fused_feature)
        return logits


#  训练与评估baseline

def train_and_evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    # 加载特征

    features_dict = torch.load("multiview_train_features.pt", map_location="cpu")

    # 划分数据集 (80/20)
    full_dataset = MultiView_Dataset(features_dict)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 初始化模型
    model = MultiView_Fusion_MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 训练模型
    epochs = 30
    print(f"\n开始训练多视图融合模型 (Epochs: {epochs})...")
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        
        for clip_feat, dino_feat, labels in train_loader:
            clip_feat = clip_feat.to(device).float()
            dino_feat = dino_feat.to(device).float()
            labels = labels.to(device).long()
            
            # 前向传播 
            optimizer.zero_grad()
            logits = model(clip_feat, dino_feat)
            
            # 依据标签计算得分
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 评估阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for clip_feat, dino_feat, labels in test_loader:
                clip_feat = clip_feat.to(device).float()
                dino_feat = dino_feat.to(device).float()
                labels = labels.to(device).long()
                
                logits = model(clip_feat, dino_feat)
                _, predicted = torch.max(logits, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}% | 耗时: {epoch_time:.2f}s")

    torch.save(model.state_dict(), "multiview_fusion_mlp.pth")
    print("\n训练完成！权重已保存至 multiview_fusion_mlp.pth")

if __name__ == "__main__":
    train_and_evaluate()