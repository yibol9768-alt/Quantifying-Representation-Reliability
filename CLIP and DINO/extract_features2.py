import torch
import clip
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import scipy.io  # 专门用于读取 Stanford Cars 的 .mat 标注文件

def extract_and_save_features():
    """
    该脚本用于从 Stanford Cars 数据集中利用 CLIP 和 DINO 两个视觉编码器
    提取多视图特征（View 1 & View 2），并将其保存到本地硬盘。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用设备: {device}")

    # ==========================================
    # 1. 加载视觉编码器模型
    # ==========================================
    
    # 加载视图 1：CLIP (ViT-B/32)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 加载视图 2：DINO (ViT-B/16)
    # 注意：首次运行会自动从 facebookresearch/dino 下载权重
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    dino_model = dino_model.to(device)
    dino_model.eval()

    # 定义 DINO 的图像预处理工作
    dino_preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ==========================================
    # 2. 解析数据集标注文件 (.mat)
    # ==========================================
    
    base_dir = "stanford_cars"
    train_img_dir = os.path.join(base_dir, "cars_train")
    # 在 devkit 目录下找到训练集标注文件
    train_annos_path = os.path.join(base_dir, "devkit", "cars_train_annos.mat")

    image_paths = []
    labels = []

    try:
        # 加载 .mat 文件
        mat_data = scipy.io.loadmat(train_annos_path)
        annotations = mat_data['annotations']
        
        # 遍历每一条标注信息
        for anno in annotations[0]:
            img_name = anno['fname'][0]
            # MATLAB 标签是 1-196，转为 PyTorch 需要的 0-195
            label = int(anno['class'][0][0]) - 1  
            
            img_path = os.path.join(train_img_dir, img_name)
            image_paths.append(img_path)
            labels.append(label)
            
        print(f"成功解析标注文件：加载了 {len(image_paths)} 张图片的路径与标签")
    except Exception as e:
        print(f"解析标注文件失败，请检查路径是否正确: {e}")
        return

    # ==========================================
    # 3. 特征提取循环
    # ==========================================

    clip_features_list = []
    dino_features_list = []
    labels_list = []

    # 冻结梯度以加快提取速度并减少显存占用
    with torch.no_grad():
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="提取特征中"):
            try:
                img = Image.open(img_path).convert("RGB")
                
                # 图像分别经过两种模型的预处理
                clip_input = clip_preprocess(img).unsqueeze(0).to(device)
                dino_input = dino_preprocess(img).unsqueeze(0).to(device)
                
                # --- 视图 1：提取并 L2 归一化 CLIP 特征 ---
                clip_feat = clip_model.encode_image(clip_input)
                clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
                
                # --- 视图 2：提取并 L2 归一化 DINO 特征 ---
                dino_feat = dino_model(dino_input)
                dino_feat = dino_feat / dino_feat.norm(dim=-1, keepdim=True)
                
                # 将张量转移到 CPU 存储以节省显存空间
                clip_features_list.append(clip_feat.cpu())
                dino_features_list.append(dino_feat.cpu())
                labels_list.append(label)
                
            except Exception as e:
                print(f"\n读取图片失败 {img_path}: {e}")

    # ==========================================
    # 4. 拼接并保存数据
    # ==========================================
    
    if len(clip_features_list) > 0:
        features_dict = {
            "clip_features": torch.cat(clip_features_list, dim=0),  # 形状: [N, 512]
            "dino_features": torch.cat(dino_features_list, dim=0),  # 形状: [N, 768]
            "labels": torch.tensor(labels_list, dtype=torch.long)   # 形状: [N]
        }
        
        save_path = "multiview_train_features.pt"
        torch.save(features_dict, save_path)
        print(f"\n任务完成！训练集多视图特征已成功保存至: {save_path}")
    else:
        print("\n未成功提取到任何特征，请检查数据集。")

if __name__ == "__main__":
    extract_and_save_features()