# Full-data vs Few-shot Scaling 实验结果

## 实验配置

**实验目的：** 验证"多模型融合中，模型数量增加时性能先升后降"的现象是 few-shot 还是 full-data 的产物

**实验设计:**
- 数据集: STL10, GTSRB, SVHN (3个)
- 模型递增: 1→2→3→4→5→6 模型
  - CLIP → +DINO → +MAE → +SigLIP → +ConvNeXt → +Data2Vec
- 设置: Few-shot (10-shot) vs Full-data
- 总计: 3 datasets × 6 steps × 2 settings = **36 runs**

**实验状态:**
- ✅ **已完成** (根据日志显示)
- ⚠️ **SSH连接超时** - 无法访问远程服务器拷贝结果

## 预期结果

### 情况A: 表征冗余是真实问题
- Few-shot 峰值在 4-5 模型
- Full-data 峰值仍在 4-5 模型
- **结论**: 表征冗余与数据量无关

### 情况B: 维度灾难是主因
- Few-shot 峰值在 4-5 模型
- Full-data 峰值上移到 5-6 或曲线一直升
- **结论**: "更多模型=更差"是 few-shot 的产物

### 情况C  两者都有影响
- Few-shot 峰值在 4-5 模型
- Full-data 峰值上移到 5-6，- **结论**: 表征冗余 + 维度灾难共同作用

## 下一步

**等待SSH连接恢复后:**

1. 拷贝结果文件到本地
2. 解析 JSON 结果
3. 生成对比曲线图
4. 撰写分析报告

## 文件位置

- 远程: `/root/autodl-tmp/feature_workspace/results/*.json`
- 远程: `/root/autodl-tmp/feature_workspace/results/*.csv`
- 本地: `/Users/liuyibo/Desktop/d/draw/`

## 监控命令

```bash
# 测试SSH连接
ssh westd "echo 'Connected'"

# 查看结果文件
ssh westd "sudo ls -lh /root/autodl-tmp/feature_workspace/results/*.json"

# 拷贝所有结果
scp -r westd:/root/autodl-tmp/feature_workspace/results/*.json /Users/liuyibo/Desktop/d/draw/

# 拷贝CSV文件
scp -r westd:/root/autodl-tmp/feature_workspace/results/*.csv /Users/liuyibo/Desktop/d/draw/
```

**当前状态**: 实验已完成，但SSH连接超时，需要等待网络恢复后拷贝结果文件。
