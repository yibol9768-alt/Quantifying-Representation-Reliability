import torch
import clip
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import scipy.io
# 提取测试集特征
def extract_test_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载clip 和 dino 视觉编码器
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    dino_model = dino_model.to(device)
    dino_model.eval()

    dino_preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 准备测试数据集
    base_dir = "stanford_cars"
    test_img_dir = os.path.join(base_dir, "cars_test") # 指向测试集图片
    
    # 指向带有真实标签的测试集标注文件
    test_annos_path = os.path.join(base_dir, "cars_test_annos_withlabels.mat")

    image_paths = []
    labels = []

    try:
        mat_data = scipy.io.loadmat(test_annos_path)
        annotations = mat_data['annotations']
        
        for anno in annotations[0]:
            img_name = anno['fname'][0]
            label = int(anno['class'][0][0]) - 1  
            
            img_path = os.path.join(test_img_dir, img_name)
            image_paths.append(img_path)
            labels.append(label)
            
        print(f"成功加载 {len(image_paths)} 张测试图片的路径与标签")
    except Exception as e:
        print(f"解析标注文件失败，请检查路径: {e}")
        return

    clip_features_list = []
    dino_features_list = []
    labels_list = []
  
    # 提取特征
    print("开始提取测试集特征")
    with torch.no_grad():
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                img = Image.open(img_path).convert("RGB")
                
                clip_input = clip_preprocess(img).unsqueeze(0).to(device)
                dino_input = dino_preprocess(img).unsqueeze(0).to(device)
                
                clip_feat = clip_model.encode_image(clip_input)
                clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
                
                dino_feat = dino_model(dino_input)
                dino_feat = dino_feat / dino_feat.norm(dim=-1, keepdim=True)
                
                clip_features_list.append(clip_feat.cpu())
                dino_features_list.append(dino_feat.cpu())
                labels_list.append(label)
                
            except Exception as e:
                print(f"读取图片失败 {img_path}: {e}")

    # 保存测试集特征
    features_dict = {
        "clip_features": torch.cat(clip_features_list, dim=0),
        "dino_features": torch.cat(dino_features_list, dim=0),
        "labels": torch.tensor(labels_list, dtype=torch.long)
    }
    
    torch.save(features_dict, "multiview_test_features.pt")
    print("测试集多视图特征已成功保存至 multiview_test_features.pt！")

if __name__ == "__main__":
    extract_test_features()