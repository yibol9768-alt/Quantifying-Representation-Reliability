"""
ImageNet train 预处理脚本：将 1000 个 .tar 文件逐一解压到对应 synset 子目录，
解压后立即删除原 .tar 文件，保证磁盘占用不翻倍。

用法:
    python cka_method/prepare_imagenet.py
    python cka_method/prepare_imagenet.py --train_dir /root/autodl-tmp/data/raw/imagenet/train
"""
import argparse
import os
import tarfile
import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="/root/autodl-tmp/data/raw/imagenet/train",
                   help="ImageNet train 目录（存放 1000 个 .tar 文件）")
    return p.parse_args()


def main():
    args = parse_args()
    train_dir = args.train_dir

    tar_files = sorted(glob.glob(os.path.join(train_dir, "*.tar")))
    if not tar_files:
        print(f"No .tar files found in {train_dir}")
        return

    total = len(tar_files)
    print(f"Found {total} tar files to extract in {train_dir}")
    print("Extracting (will delete each tar after extraction to save disk space)...\n")

    for i, tar_path in enumerate(tar_files, 1):
        wnid = os.path.basename(tar_path).replace(".tar", "")
        target_dir = os.path.join(train_dir, wnid)

        # 如果目录已存在且有内容，跳过
        if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 0:
            print(f"[{i}/{total}] {wnid}: already extracted, skipping")
            # 删除冗余的 tar 文件（如果还存在）
            if os.path.exists(tar_path):
                os.remove(tar_path)
            continue

        os.makedirs(target_dir, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(target_dir)
            os.remove(tar_path)
            print(f"[{i}/{total}] {wnid}: extracted and tar deleted")
        except Exception as e:
            print(f"[{i}/{total}] {wnid}: ERROR - {e}")

    # 统计结果
    synset_dirs = [d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))]
    print(f"\nDone! {len(synset_dirs)} synset directories in {train_dir}")

    # 验证可以用 ImageFolder 加载
    try:
        import torchvision.datasets as tv_datasets
        import torchvision.transforms as T
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        ds = tv_datasets.ImageFolder(train_dir, transform=transform)
        print(f"ImageFolder verification: {len(ds)} images, {len(ds.classes)} classes")
    except Exception as e:
        print(f"ImageFolder verification failed: {e}")


if __name__ == "__main__":
    main()
