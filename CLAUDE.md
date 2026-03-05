# 项目开发指南

本文档定义项目开发的规范和工作流程，确保代码质量和项目文档的同步更新。

---

## 代码修改规范

### 每次修改代码时，必须检查以下内容：

1. **README 同步检查**
   - 修改了功能/脚本 → 更新 README 中的对应说明
   - 新增了文件 → 更新项目结构
   - 修改了参数 → 更新配置说明
   - 修改了依赖 → 更新环境配置部分
   - 发现错误 → 修正 README 中的错误描述

2. **代码质量检查**
   - 代码是否有清晰的注释
   - 函数是否有 docstring 说明
   - 变量命名是否清晰易懂
   - 是否有明显的 bug 或逻辑错误

3. **测试验证**
   - 修改后的代码能否正常运行
   - 是否需要更新测试脚本

---

## 任务完成规范

### 完成一项任务后，必须执行以下步骤：

#### 1. 代码提交

```bash
# 查看修改内容
git status
git diff

# 添加修改的文件
git add <files>
# 或添加所有修改
git add .

# 提交（使用清晰的提交信息）
git commit -m "<type>: <description>"

# 提交类型示例：
# feat: 新增功能
# fix: 修复 bug
# docs: 更新文档
# refactor: 代码重构
# style: 代码格式调整
# test: 测试相关
# chore: 构建/工具链相关
```

#### 2. 推送到 GitHub

```bash
# 推送到远程仓库
git push origin main

# 如果是新分支，使用：
git push -u origin <branch-name>
```

#### 3. 验证推送

访问 GitHub 仓库页面，确认：
- 代码已成功推送
- README 显示正确
- 文件结构完整

---

## 项目状态追踪

### 当前任务列表

- [ ] 完善项目文档
- [ ] 运行双视图实验并记录结果
- [ ] 运行三视图实验并记录结果
- [ ] 对比不同方案的性能
- [ ] 添加单模型基线实验
- [ ] 整理实验结果和图表

---

## Git 工作流建议

### 分支策略

```
main (主分支，稳定版本)
  ├── feature-xxx (功能开发分支)
  ├── fix-xxx (bug 修复分支)
  └── docs-xxx (文档更新分支)
```

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

示例：
```
feat(extraction): add MAE feature extraction support

- Added MAE model loading from Hugging Face
- Implemented MAE-specific preprocessing pipeline
- Updated README with new feature extraction steps

Closes #1
```

---

## 代码修改检查清单

在提交代码前，请确认：

- [ ] 代码可以正常运行
- [ ] README 已同步更新（如需要）
- [ ] 添加了必要的注释
- [ ] 没有调试代码残留（print、测试代码等）
- [ ] 敏感信息已移除（路径、密钥等）
- [ ] Git 提交信息清晰明确
- [ ] 已推送到 GitHub

---

## 注意事项

1. **语言规范** ⚠️
   - **仅 README 使用中文**，方便中文用户理解
   - **其他地方全部使用英文**：
     - 代码注释
     - 变量/函数名
     - 文件名（除 README.md）
     - Git 提交信息
     - 配置文件

2. **系统代理配置** (国内网络加速)

   在 AutoDL 等国内服务器上运行时，需要配置镜像加速：

   ```bash
   # HuggingFace 镜像 (已集成到代码中)
   export HF_ENDPOINT=https://hf-mirror.com

   # GitHub 加速 (如需从 GitHub 下载模型)
   export GITHUB=https://github.com.cn

   # PyPI 镜像
   export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

   # 如果有 HTTP/HTTPS 代理，设置：
   # export http_proxy=http://proxy.example.com:port
   # export https_proxy=http://proxy.example.com:port
   ```

3. **不要提交大数据文件**
   - 特征文件 (`*.pt`) 应加入 `.gitignore`
   - 模型权重文件应加入 `.gitignore`
   - 数据集文件不要提交

4. **敏感信息保护**
   - 不要提交包含本地绝对路径的代码
   - 使用相对路径或配置文件

5. **文档同步**
   - README 是项目的主要文档
   - 修改代码时务必考虑是否需要更新文档

---

## 快捷命令

```bash
# 完整的提交流程
git add .
git commit -m "docs: update README"
git push origin main

# 查看最近 3 次提交
git log -3 --oneline

# 查看当前分支状态
git status

# 撤销未提交的修改
git checkout -- <file>
```
