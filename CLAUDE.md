# 项目开发规范

本文档定义项目开发的规范和工作流程。

---

## 代码修改规范

每次修改代码时，必须检查：
1. README 是否需要同步更新
2. 代码是否有清晰注释
3. 变量命名是否清晰
4. 修改后的代码能否正常运行

---

## Git 提交规范

```bash
git add <files>
git commit -m "<type>: <description>"

# 类型：
# feat: 新增功能
# fix: 修复 bug
# docs: 更新文档
# refactor: 代码重构
```

---

## AutoDL 环境配置

### 数据盘位置

```bash
# 持久化数据盘（推荐用于存放数据集和模型）
/root/autodl-fs/data/

# 临时盘（重启后清空）
/root/autodl-tmp/
```

### 网络加速

```bash
# 学术网络加速（访问 GitHub/HuggingFace）
source /etc/network_turbo

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 常用路径

```bash
# 项目目录
cd /root/autodl-fs/Quantifying-Representation-Reliability

# 数据集存放
mkdir -p /root/autodl-fs/data/datasets

# 特征文件输出
mkdir -p /root/autodl-fs/Quantifying-Representation-Reliability/features

# 模型检查点
mkdir -p /root/autodl-fs/Quantifying-Representation-Reliability/outputs/checkpoints
```

---

## 分支说明

- `main` - 主分支，稳定版本
- `feature/comm-fusion` - COMM 融合方法开发分支

---

## 语言规范

- **README**: 中文
- **代码注释**: 英文
- **变量/函数名**: 英文
- **Git 提交**: 英文

---

## 不要提交

- `data/` - 数据集文件
- `logs/` - 日志文件
- `features/*.pt` - 特征文件（可重新生成）
- `outputs/checkpoints/*.pth` - 模型权重（可重新训练）

在 `.gitignore` 中配置：
```
data/
logs/
features/*.pt
outputs/checkpoints/*.pth
```
