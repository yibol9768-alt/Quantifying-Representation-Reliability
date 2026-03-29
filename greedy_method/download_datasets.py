"""
下载 GTSRB、DTD、Pets 三个数据集到 /root/autodl-tmp/data/raw/
"""
import os, sys, time
import torchvision.datasets as D

DATA_ROOT = "/root/autodl-tmp/data/raw"

datasets = [
    ("GTSRB",          "gtsrb"),
    ("DTD",            "dtd"),
    ("OxfordIIITPet",  "pets"),
]

for display_name, folder in datasets:
    dest = os.path.join(DATA_ROOT, folder)
    os.makedirs(dest, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  正在下载: {display_name}  →  {dest}")
    print(f"{'='*55}")
    t0 = time.time()
    try:
        if folder == "gtsrb":
            D.GTSRB(dest, split="train", download=True)
            D.GTSRB(dest, split="test",  download=True)
        elif folder == "dtd":
            D.DTD(dest, split="train", download=True)
            D.DTD(dest, split="val",   download=True)
            D.DTD(dest, split="test",  download=True)
        elif folder == "pets":
            D.OxfordIIITPet(dest, split="trainval", download=True)
            D.OxfordIIITPet(dest, split="test",     download=True)
        elapsed = time.time() - t0
        print(f"  ✓ {display_name} 下载完成！耗时 {elapsed:.1f}s")
    except Exception as e:
        print(f"  ✗ {display_name} 下载失败: {e}", file=sys.stderr)

print("\n\n所有数据集下载完毕，存储位置：")
for _, folder in datasets:
    path = os.path.join(DATA_ROOT, folder)
    if os.path.exists(path):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(path) for f in files
        ) / 1024**2
        print(f"  {path}  ({size:.0f} MB)")
