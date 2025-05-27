# 信用卡违约预测项目

这个项目使用机器学习方法预测不同地区的信用卡违约情况，并提供API服务。

## 项目结构

```
.
├── notebooks/                 # Jupyter notebooks用于探索性分析
│   ├── taiwan.ipynb
│   ├── Australia.ipynb
│   └── german.ipynb
├── src/                      # 源代码目录
│   ├── data/                 # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── data_loader.py    # 数据加载和预处理
│   │   └── feature_engineering.py
│   ├── models/               # 模型相关代码
│   │   ├── __init__.py
│   │   ├── train.py         # 模型训练
│   │   ├── evaluate.py      # 模型评估
│   │   └── predict.py       # 模型预测
│   └── api/                  # FastAPI服务
│       ├── __init__.py
│       ├── main.py          # API主程序
│       └── schemas.py       # API数据模型
├── tests/                    # 单元测试
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── configs/                  # 配置文件
│   ├── model_config.yaml    # 模型配置
│   └── api_config.yaml      # API配置
├── models/                   # 保存训练好的模型
│   └── .gitkeep
├── data/                     # 数据目录
│   └── .gitkeep
├── requirements.txt          # 项目依赖
├── requirements-dev.txt      # 开发环境依赖
├── setup.py                 # 项目安装配置
└── README.md                # 项目说明文档
```

## 环境设置

1. 克隆仓库：
```bash
git clone [repository-url]
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发环境
```

## 常见问题

### Q: 模型文件未找到
**A**: 确保先运行训练脚本生成模型文件

### Q: W&B登录失败
**A**: 
1. 检查API密钥是否正确
2. 确保网络连接正常
3. 可以先不使用W&B：`python train_model.py`

### Q: API启动失败
**A**: 检查8000端口是否被占用，或修改端口号

### Q: 内存不足
**A**: 可以在配置文件中减少数据样本数量

## 使用说明

### 方式1: 快速开始（不使用W&B）

适合新手快速体验项目功能：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python train_model.py

# 3. 启动API服务
python start_api.py
```

### 方式2: 完整版（使用W&B实验跟踪）

推荐用于正式开发和实验管理：

#### 2.1 配置W&B

1. **注册W&B账户**：
   - 访问 https://wandb.ai/ 注册免费账户

2. **获取API密钥**：
   - 登录后访问 https://wandb.ai/authorize
   - 复制您的API密钥

3. **登录W&B**：
   ```bash
   wandb login
   # 粘贴您的API密钥
   ```

4. **修改配置文件**：
   编辑 `configs/model_config.yaml`，设置您的W&B用户名：
   ```yaml
   wandb:
     project: "credit-card-default-prediction"
     entity: "your-username"  # 替换为您的W&B用户名
   ```

#### 2.2 运行完整训练

```bash
# 使用W&B进行实验跟踪
python train_model.py --use-wandb

# 或者直接运行W&B专用脚本
python train_with_wandb.py
```

#### 2.3 W&B功能

训练完成后，您可以在W&B dashboard中查看：
- 📊 **实验指标**：accuracy、F1-score、ROC-AUC等
- 📈 **损失曲线**：训练过程可视化
- 🎯 **模型对比**：不同实验的性能对比
- 💾 **模型Artifacts**：自动保存和版本管理
- 🔄 **实验复现**：完整的代码和环境记录

### API服务使用

```bash
# 启动API服务
python start_api.py
```

服务启动后可访问：
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

#### API测试示例

```bash
# 健康检查
curl http://localhost:8000/health

# 单个预测
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "LIMIT_BAL": 20000,
         "SEX": 2,
         "EDUCATION": 2,
         "MARRIAGE": 1,
         "AGE": 24,
         "PAY_0": 2,
         "PAY_2": 2,
         "PAY_3": -1,
         "PAY_4": -1,
         "PAY_5": -2,
         "PAY_6": -2,
         "BILL_AMT1": 3913,
         "BILL_AMT2": 3102,
         "BILL_AMT3": 689,
         "BILL_AMT4": 0,
         "BILL_AMT5": 0,
         "BILL_AMT6": 0,
         "PAY_AMT1": 0,
         "PAY_AMT2": 689,
         "PAY_AMT3": 0,
         "PAY_AMT4": 0,
         "PAY_AMT5": 0,
         "PAY_AMT6": 0
       }
     }'
```

## 项目特点

- 使用Weights & Biases进行实验跟踪和模型管理
- 支持模型版本控制和实验复现
- 提供RESTful API服务
- 包含完整的测试套件
- 支持模型部署和监控

## 开发指南

1. 代码风格遵循PEP 8规范
2. 所有新功能需要添加单元测试
3. 使用pre-commit hooks确保代码质量
4. 遵循语义化版本控制 