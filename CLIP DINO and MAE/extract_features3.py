import torch
import clip
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import scipy.io  # 专门用于读取 Stanford Cars 的 .mat 标注文件
from transformers import ViTMAEModel, AutoImageProcessor # 引用hugging face 的 transformers 库来加载MAE

def extract_and_save_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载三大视觉编码器 
    # clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    # dino
    # 先从 github 下载权重
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    dino_model = dino_model.to(device)
    dino_model.eval()

    # dino的图像预处理工作
    dino_preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # 加载 MAE 专用的图像预处理器和模型权重 (Base版本)
    mae_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    mae_model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device)
    mae_model.eval()
    
    # 准备数据集并精准解析 Stanford Cars .mat 文件
   
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
            # MATLAB 标签是 1-196，PyTorch 必须是 0-195，所以要减 1
            label = int(anno['class'][0][0]) - 1  
            
            img_path = os.path.join(train_img_dir, img_name)
            image_paths.append(img_path)
            labels.append(label)
            
        print(f"成功加载 {len(image_paths)} 张图片的路径与标签")
    except Exception as e:
        print(f"解析标注文件失败，请检查路径: {e}")
        return

    clip_features_list = []
    dino_features_list = []
    mae_features_list = []
    labels_list = []
  
    # 冻结梯度 特征提取
    with torch.no_grad():
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                img = Image.open(img_path).convert("RGB")
                
                # 分别经过三种预处理
                clip_input = clip_preprocess(img).unsqueeze(0).to(device)
                dino_input = dino_preprocess(img).unsqueeze(0).to(device)
                mae_input = mae_processor(images=img, return_tensors="pt").to(device)
                
                # 视图 1：提取并归一化 CLIP 特征 [1, 512]
                clip_feat = clip_model.encode_image(clip_input)
                clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
                
                # 视图 2：提取并归一化 DINO 特征 [1, 768]
                dino_feat = dino_model(dino_input)
                dino_feat = dino_feat / dino_feat.norm(dim=-1, keepdim=True)

                # 视图 3：MAE 提取 (768维)
                mae_output = mae_model(**mae_input)
                # MAE 输出的是所有图像块的特征，取第 0 个位置的 CLS Token 作为全局特征
                mae_feat = mae_output.last_hidden_state[:, 0, :]
                mae_feat = mae_feat / mae_feat.norm(dim=-1, keepdim=True) # 进行 L2 归一化
                
                # 转移到 CPU 保存以节省显存
                clip_features_list.append(clip_feat.cpu())
                dino_features_list.append(dino_feat.cpu())
                mae_features_list.append(mae_feat.cpu())
                labels_list.append(label)
                
            except Exception as e:
                print(f"读取图片失败 {img_path}: {e}")

    
    # 拼接并保存到本地硬盘
    
    features_dict = {
        "clip_features": torch.cat(clip_features_list, dim=0),  # 形状: [8144, 512]
        "dino_features": torch.cat(dino_features_list, dim=0),  # 形状: [8144, 768]
        "mae_features": torch.cat(mae_features_list, dim=0),    # 形状: [8144, 768]
        "labels": torch.tensor(labels_list, dtype=torch.long)   # 形状: [8144]
    }
    
    torch.save(features_dict, "multiview3_train_features.pt")
    print("训练集多视图特征已成功保存至 multiview3_train_features.pt！")

if __name__ == "__main__":
    extract_and_save_features()