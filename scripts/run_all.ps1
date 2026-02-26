# 一键运行脚本 (Windows)
#
# 使用前:
#   conda create -n repreli python=3.10
#   conda activate repreli
#   pip install -r requirements.txt
#
# 运行:
#   .\scripts\run_all.ps1

param([string]$Config = "configs/default.yaml")

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host "Heterogeneous Model Evaluation Pipeline"
Write-Host "=============================================="
Write-Host ""

# 检查 Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python not found"
    exit 1
}

# 检查依赖
Write-Host "Checking dependencies..."
pip show torch transformers numpy scipy scikit-learn > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..."
    pip install -r requirements.txt
}

# 创建目录
New-Item -ItemType Directory -Force -Path data, features, results, model_cache | Out-Null

# ============================================================
# Step 1: 下载预训练模型 (不需要训练！)
# ============================================================
Write-Host ""
Write-Host "[1/3] Downloading pretrained models..."
Write-Host "----------------------------------------------"
python scripts/download_models.py --family clip dinov2 mae --cache_dir ./model_cache

# ============================================================
# Step 2: 提取特征
# ============================================================
Write-Host ""
Write-Host "[2/3] Extracting features..."
Write-Host "----------------------------------------------"

$models = python -c "import yaml; c=yaml.safe_load(open('$Config')); print(' '.join(c['models']))"
$dataset = python -c "import yaml; c=yaml.safe_load(open('$Config')); print(c['dataset'])"

Write-Host "Models: $models"
Write-Host "Dataset: $dataset"

python scripts/extract_features.py `
    --models $models.Split() `
    --dataset $dataset `
    --output ./features `
    --split both

# ============================================================
# Step 3: 训练线性头 + 计算 NC + 评估
# ============================================================
Write-Host ""
Write-Host "[3/3] Running evaluation..."
Write-Host "----------------------------------------------"
Write-Host "Training linear heads and computing NC scores..."
Write-Host ""

python scripts/run_evaluation.py --config $Config

# ============================================================
# 结果
# ============================================================
Write-Host ""
Write-Host "=============================================="
Write-Host "Done!"
Write-Host "=============================================="
Write-Host ""
Write-Host "Results:"
Get-ChildItem results/
