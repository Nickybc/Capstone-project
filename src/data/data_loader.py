"""数据加载和预处理模块."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging
from ucimlrepo import fetch_ucirepo
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
import wandb
import yaml
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类，用于处理信用卡违约数据."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        初始化数据加载器.
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_names = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'AGE',  # X1 - X4
            'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3',   # X5 - X8
            'PAY_4', 'PAY_5', 'PAY_6',               # X9 - X11
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',   # X12 - X14
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',   # X15 - X17
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',      # X18 - X20
            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'       # X21 - X23
        ]
        
        self.X = None
        self.y = None
        self.feature_importance = None
        
    def load_taiwan_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载台湾信用卡数据集.
        
        Returns:
            特征数据和目标变量
        """
        logger.info("正在下载台湾信用卡数据集...")
        
        # 从UCI仓库获取数据
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features
        y = dataset.data.targets
        
        # 设置正确的特征名称
        X.columns = self.feature_names
        
        # 记录数据集信息
        logger.info(f"数据集形状: {X.shape}")
        logger.info(f"目标变量分布:\n{y.value_counts()}")
        
        # 数据基本信息检查
        logger.info("检查缺失值...")
        missing_values = X.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_values[missing_values > 0]}")
        else:
            logger.info("数据无缺失值")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                   k: int = 17) -> Tuple[pd.DataFrame, List[str]]:
        """
        使用信息增益计算特征重要性并选择top-k特征.
        
        Args:
            X: 特征数据
            y: 目标变量
            k: 选择的特征数量
            
        Returns:
            选择的特征数据和特征名称列表
        """
        logger.info("计算特征重要性（信息增益）...")
        
        # 使用互信息作为信息增益的近似
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y.values.ravel())
        
        # 获取特征重要性分数
        feature_scores = pd.Series(
            selector.scores_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        # 获取选择的特征名称
        selected_features = X.columns[selector.get_support()].tolist()
        
        # 创建选择后的DataFrame
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
        
        logger.info(f"选择了 {k} 个特征")
        logger.info(f"特征重要性排序:\n{feature_scores}")
        
        self.feature_importance = feature_scores
        
        return X_selected_df, selected_features
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.3, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series]:
        """
        划分训练集和测试集.
        
        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练集和测试集的特征和目标变量
        """
        logger.info(f"划分数据集，测试集比例: {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"训练集大小: {X_train.shape[0]}")
        logger.info(f"测试集大小: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def save_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series,
                  output_dir: str = "data") -> None:
        """
        保存处理后的数据.
        
        Args:
            X_train, X_test, y_train, y_test: 训练和测试数据
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        logger.info(f"数据已保存到 {output_dir} 目录")
    
    def log_to_wandb(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        将数据信息记录到Weights & Biases.
        
        Args:
            X: 特征数据
            y: 目标变量
        """
        if wandb.run is not None:
            # 记录数据集统计信息
            wandb.log({
                "dataset/num_samples": len(X),
                "dataset/num_features": X.shape[1],
                "dataset/positive_class_ratio": y.sum() / len(y),
                "dataset/feature_names": X.columns.tolist()
            })
            
            # 记录特征重要性
            if self.feature_importance is not None:
                importance_dict = {
                    f"feature_importance/{feature}": score
                    for feature, score in self.feature_importance.items()
                }
                wandb.log(importance_dict)
    
    def process_pipeline(self, k_features: int = 17) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                            pd.Series, pd.Series]:
        """
        完整的数据处理流水线.
        
        Args:
            k_features: 选择的特征数量
            
        Returns:
            处理后的训练和测试数据
        """
        # 加载数据
        X, y = self.load_taiwan_data()
        
        # 特征选择
        X_selected, selected_features = self.calculate_feature_importance(X, y, k_features)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = self.split_data(
            X_selected, y,
            test_size=1 - self.config['data']['train_size'],
            random_state=self.config['data']['random_state']
        )
        
        # 保存数据
        self.save_data(X_train, X_test, y_train, y_test)
        
        # 记录到wandb
        self.log_to_wandb(X_selected, y)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据加载器并运行处理流水线
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.process_pipeline()
    
    print(f"数据处理完成！")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}") 