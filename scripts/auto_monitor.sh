#!/bin/bash
# 自动监控实验并上传结果到 GitHub

PROJECT_DIR="/home/claudeuser/Quantifying-Representation-Reliability"
cd "$PROJECT_DIR"

echo "=== 自动监控脚本 ==="
echo "实验运行在 tmux 会话 'exp' 中"
echo ""

# 检查 tmux 会话是否存在
if ! tmux list-sessions 2>/dev/null | grep -q "exp"; then
    echo "❌ tmux 会话 'exp' 不存在"
    echo "请先启动实验: tmux new-session -d -s exp 'python3 scripts/run_experiments.py'"
    exit 1
fi

echo "✓ tmux 会话运行中"
echo ""

# 函数：检查进度
check_progress() {
    local features=$(ls -1 features/*.pt 2>/dev/null | wc -l)
    local checkpoints=$(ls -1 outputs/checkpoints/*.pth 2>/dev/null | wc -l)
    local total=$((features + checkpoints))
    local percent=$((total * 100 / 39))

    echo "=== 当前进度 ==="
    echo "特征文件: $features/18"
    echo "模型文件: $checkpoints/21"
    echo "总进度: $total/39 ($percent%)"
    echo ""

    # 按数据集分组显示
    echo "--- 特征提取详情 ---"
    for ds in cifar100 flowers102 pets; do
        local count=$(ls -1 features/${ds}_*.pt 2>/dev/null | wc -l)
        printf "%-12s %d/6\n" "$ds:" "$count"
    done
    echo ""
}

# 函数：检查 GPU 状态
check_gpu() {
    echo "=== GPU 状态 ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null || echo "GPU: 不可用"
    echo ""
}

# 函数：收集结果
collect_results() {
    echo "=== 收集实验结果 ==="

    # 创建结果文件
    cat > RESULTS_TEMP.md << 'EOF'
# 多视图预训练模型融合实验结果

实验日期: $(date '+%Y-%m-%d %H:%M:%S')
GPU: NVIDIA GeForce RTX 2080 Ti

## 数据集

| 数据集 | 类别数 | 训练集 | 测试集 | 类型 |
|--------|--------|--------|--------|------|
| CIFAR-100 | 100 | 50,000 | 10,000 | 通用 |
| Flowers-102 | 102 | 1,020 | 6,149 | 细粒度 |
| Oxford-IIIT Pets | 37 | 3,680 | 3,669 | 细粒度 |

## 实验结果

### 单模型准确率 (%)

| 数据集 | CLIP | DINO | MAE |
|--------|------|------|------|
EOF

    # 收集训练日志中的准确率
    for log_file in logs/*_train.log; do
        if [ -f "$log_file" ]; then
            local dataset=$(basename "$log_file" | cut -d_ -f1)
            local model=$(basename "$log_file" | cut -d_ -f2)

            # 提取测试准确率
            local test_log="logs/${dataset}_${model}_test.log"
            if [ -f "$test_log" ]; then
                local acc=$(grep -oE "Test Acc: [0-9.]+" "$test_log" | tail -1 | cut -d' ' -f3 2>/dev/null)
                if [ -z "$acc" ]; then
                    acc=$(grep -oE "accuracy: [0-9]+.[0-9]+" "$test_log" | tail -1 | cut -d' ' -f2 2>/dev/null)
                fi
                if [ -n "$acc" ]; then
                    echo "| $dataset | $acc |" >> RESULTS_TEMP.md
                fi
            fi
        fi
    done

    echo "" >> RESULTS_TEMP.md
    echo "### 双模型融合" >> RESULTS_TEMP.md
    echo "" >> RESULTS_TEMP.md
    echo "| 数据集 | CLIP+DINO | CLIP+MAE | DINO+MAE |" >> RESULTS_TEMP.md
    echo "|--------|----------|---------|----------|" >> RESULTS_TEMP.md

    # 收集融合模型结果
    for log_file in logs/*_fusion.log; do
        if [ -f "$log_file" ]; then
            local dataset=$(basename "$log_file" | cut -d_ -f1)
            local models=$(basename "$log_file" | cut -d_ -f2- | sed 's/_fusion//')
            local acc=$(grep -oE "Test Acc: [0-9.]+" "$log_file" | tail -1 | cut -d' ' -f3 2>/dev/null)
            if [ -n "$acc" ]; then
                echo "| $dataset | $acc |" >> RESULTS_TEMP.md
            fi
        fi
    done

    echo "" >> RESULTS_TEMP.md
    echo "### 三模型融合 (CLIP+DINO+MAE)" >> RESULTS_TEMP.md
    echo "" >> RESULTS_TEMP.md
    echo "| 数据集 | 准确率 |" >> RESULTS_TEMP.md
    echo "|--------|--------|" >> RESULTS_TEMP.md

    # 收集三模型融合结果
    for log_file in logs/*_3model_fusion.log logs/*_clip_dino_mae_fusion.log; do
        if [ -f "$log_file" ]; then
            local dataset=$(basename "$log_file" | cut -d_ -f1)
            local acc=$(grep -oE "Test Acc: [0-9.]+" "$log_file" | tail -1 | cut -d' ' -f3 2>/dev/null)
            if [ -n "$acc" ]; then
                echo "| $dataset | $acc |" >> RESULTS_TEMP.md
            fi
        fi
    done

    mv RESULTS_TEMP.md RESULTS.md
    echo "✓ 结果已保存到 RESULTS.md"
    cat RESULTS.md
}

# 函数：上传到 GitHub
upload_to_github() {
    echo "=== 上传到 GitHub ==="

    # 配置 git
    git config user.name "Claude Auto" 2>/dev/null
    git config user.email "claude@anthropic.com" 2>/dev/null

    # 添加修改
    git add RESULTS.md 2>/dev/null
    git commit -m "docs: update experimental results

- Update classification accuracy on cifar100, flowers102, pets
- Single model, dual fusion, and triple fusion results
- GPU-accelerated training

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" 2>/dev/null

    # 推送到 GitHub
    if git push origin main 2>&1; then
        echo "✓ 已上传到 GitHub"
    else
        echo "❌ 上传失败，可能需要认证"
        echo "请手动运行: git push origin main"
    fi
}

# 函数：检查实验是否完成
is_experiment_done() {
    local features=$(ls -1 features/*.pt 2>/dev/null | wc -l)
    local checkpoints=$(ls -1 outputs/checkpoints/*.pth 2>/dev/null | wc -l)
    local total=$((features + checkpoints))

    # 假设所有任务完成
    if [ $total -ge 39 ]; then
        return 0
    fi
    return 1
}

# 主循环
echo "开始监控实验..."
echo "每 60 秒检查一次进度"
echo ""

while true; do
    check_progress
    check_gpu

    if is_experiment_done; then
        echo ""
        echo "🎉 实验完成！"
        echo ""
        collect_results
        upload_to_github
        break
    fi

    echo "等待 60 秒..."
    sleep 60
done

echo ""
echo "=== 监控结束 ==="
