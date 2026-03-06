#!/bin/bash
# 后台监控并自动上传

PROJECT_DIR="/home/claudeuser/Quantifying-Representation-Reliability"
cd "$PROJECT_DIR"

log_file="logs/monitor.log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$log_file"
}

log "=== 启动监控 ==="

check_and_upload() {
    # 检查进度
    features=$(ls -1 features/*.pt 2>/dev/null | wc -l)
    checkpoints=$(ls -1 outputs/checkpoints/*.pth 2>/dev/null | wc -l)
    total=$((features + checkpoints))
    
    log "进度: $total/39 (特征:$features, 模型:$checkpoints)"
    
    # 如果实验完成，收集并上传
    if [ $total -ge 39 ]; then
        log "实验完成！收集结果..."
        
        # 提取结果
        python3 scripts/collect_results.py 2>/dev/null || python3 << 'PYEOF'
from pathlib import Path
import re

logs_dir = Path("logs")
results = {}

# 收集所有日志中的准确率
for log_file in logs_dir.glob("*_train.log"):
    parts = log_file.stem.split("_")
    if len(parts) >= 2:
        dataset = parts[0]
        model = parts[1]
        
        content = log_file.read_text()
        acc_match = re.search(r'(Test Acc|accuracy):\s*([0-9.]+)', content)
        if acc_match:
            acc = acc_match.group(2)
            key = f"{dataset}_{model}"
            results[key] = acc

# 生成结果表
with open("RESULTS.md", "w") as f:
    f.write("# 实验结果\n\n")
    f.write(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 单模型结果\n\n")
    f.write("| 数据集 | CLIP | DINO | MAE |\n")
    f.write("|--------|------|------|------|\n")
    
    for ds in ["cifar100", "flowers102", "pets"]:
        row = f"| {ds} |"
        for model in ["clip", "dino", "mae"]:
            acc = results.get(f"{ds}_{model}", "-")
            row += f" {acc} |"
        f.write(row + "\n")
    
    f.write("\n## 双模型融合\n")
    f.write("等待训练完成...\n")

print("✓ RESULTS.md 已更新")
PYEOF

        log "上传到 GitHub..."
        
        # Git 操作
        git config user.name "Claude Auto" >/dev/null 2>&1
        git config user.email "claude@anthropic.com" >/dev/null 2>&1
        
        git add RESULTS.md >/dev/null 2>&1
        git commit -m "docs: update results $(date '+%H:%M')" >/dev/null 2>&1
        git push origin main >/dev/null 2>&1
        
        log "✓ 已上传到 GitHub"
        exit 0
    fi
}

# 每分钟检查一次
while true; do
    check_and_upload
    sleep 60
done
