import subprocess
import sys
# 提取训练集特征 -> 训练模型 -> 提取测试集特征 -> 输出测试结果
scripts_to_run = [
    "extract_features2.py",  # 提取特征 (CLIP, DINO, MAE)
    "train_baseline2.py",    # 训练基线模型/进行特征融合
    "extract_test2.py",      # 提取测试集特征
    "model_test2.py"         # 测试并输出最终结果
]

def main():
    
    for script in scripts_to_run:
        
        # 使用当前环境的 Python 解释器运行脚本
        result = subprocess.run([sys.executable, script])
        # 检查上一个脚本是否成功运行
        if result.returncode != 0:
            print(f"\n运行 '{script}' 时发生错误")
            sys.exit(result.returncode)
            
        print(f"'{script}' 运行完成\n")


if __name__ == "__main__":
    main()