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

4. 设置Weights & Biases：
```bash
wandb login
```

## 使用说明

### 模型训练

1. 数据预处理：
```bash
python src/data/data_loader.py
```

2. 模型训练：
```bash
python src/models/train.py
```

### API服务

启动FastAPI服务：
```bash
uvicorn src.api.main:app --reload
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