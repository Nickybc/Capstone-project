"""特征工程模块."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """特征工程类，用于数据预处理和特征转换."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        初始化特征工程器.
        
        Args:
            scaler_type: 缩放器类型 ('standard' 或 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type 必须是 'standard' 或 'minmax'")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        拟合特征工程器.
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            
        Returns:
            self
        """
        logger.info(f"拟合特征工程器，使用 {self.scaler_type} 缩放器")
        
        self.feature_names = X.columns.tolist()
        
        # 拟合缩放器
        self.scaler.fit(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换特征数据.
        
        Args:
            X: 特征数据
            
        Returns:
            转换后的特征数据
        """
        if self.scaler is None:
            raise ValueError("特征工程器未拟合，请先调用 fit 方法")
        
        # 确保列顺序一致
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        # 应用缩放
        X_scaled = self.scaler.transform(X)
        
        # 转换为DataFrame
        X_transformed = pd.DataFrame(
            X_scaled, 
            columns=self.feature_names,
            index=X.index
        )
        
        logger.info(f"完成特征转换，形状: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        拟合并转换特征数据.
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            
        Returns:
            转换后的特征数据
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        逆转换特征数据.
        
        Args:
            X: 转换后的特征数据
            
        Returns:
            原始特征数据
        """
        if self.scaler is None:
            raise ValueError("特征工程器未拟合，请先调用 fit 方法")
        
        X_inverse = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(
            X_inverse,
            columns=self.feature_names,
            index=X.index
        )
    
    def get_feature_names(self) -> list:
        """
        获取特征名称列表.
        
        Returns:
            特征名称列表
        """
        return self.feature_names
    
    def save_scaler(self, filepath: str) -> None:
        """
        保存缩放器.
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info(f"缩放器已保存到: {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        加载缩放器.
        
        Args:
            filepath: 加载路径
        """
        self.scaler = joblib.load(filepath)
        logger.info(f"缩放器已从 {filepath} 加载")
    
    def get_scaling_params(self) -> Dict[str, Any]:
        """
        获取缩放参数.
        
        Returns:
            缩放参数字典
        """
        if self.scaler is None:
            return {}
        
        params = {}
        
        if hasattr(self.scaler, 'mean_'):
            params['mean'] = self.scaler.mean_
        if hasattr(self.scaler, 'scale_'):
            params['scale'] = self.scaler.scale_
        if hasattr(self.scaler, 'min_'):
            params['min'] = self.scaler.min_
        if hasattr(self.scaler, 'data_min_'):
            params['data_min'] = self.scaler.data_min_
        if hasattr(self.scaler, 'data_max_'):
            params['data_max'] = self.scaler.data_max_
        
        return params


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """分类特征编码器."""
    
    def __init__(self, categorical_features: list):
        """
        初始化分类编码器.
        
        Args:
            categorical_features: 分类特征列表
        """
        self.categorical_features = categorical_features
        self.encoders = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        拟合编码器.
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            
        Returns:
            self
        """
        for feature in self.categorical_features:
            if feature in X.columns:
                # 简单的标签编码
                unique_values = X[feature].unique()
                self.encoders[feature] = {val: idx for idx, val in enumerate(unique_values)}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换分类特征.
        
        Args:
            X: 特征数据
            
        Returns:
            转换后的特征数据
        """
        X_transformed = X.copy()
        
        for feature, encoder in self.encoders.items():
            if feature in X_transformed.columns:
                X_transformed[feature] = X_transformed[feature].map(encoder)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        拟合并转换分类特征.
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            
        Returns:
            转换后的特征数据
        """
        return self.fit(X, y).transform(X)


def create_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    创建衍生特征.
    
    Args:
        X: 原始特征数据
        
    Returns:
        包含衍生特征的数据
    """
    X_derived = X.copy()
    
    # 计算账单金额的统计特征
    bill_cols = [col for col in X.columns if 'BILL_AMT' in col]
    if bill_cols:
        X_derived['BILL_AMT_MEAN'] = X[bill_cols].mean(axis=1)
        X_derived['BILL_AMT_STD'] = X[bill_cols].std(axis=1)
        X_derived['BILL_AMT_MAX'] = X[bill_cols].max(axis=1)
        X_derived['BILL_AMT_MIN'] = X[bill_cols].min(axis=1)
    
    # 计算支付金额的统计特征
    pay_cols = [col for col in X.columns if 'PAY_AMT' in col]
    if pay_cols:
        X_derived['PAY_AMT_MEAN'] = X[pay_cols].mean(axis=1)
        X_derived['PAY_AMT_STD'] = X[pay_cols].std(axis=1)
        X_derived['PAY_AMT_MAX'] = X[pay_cols].max(axis=1)
        X_derived['PAY_AMT_MIN'] = X[pay_cols].min(axis=1)
    
    # 计算还款状态的统计特征
    pay_status_cols = [col for col in X.columns if col.startswith('PAY_') and col != 'PAY_0']
    if pay_status_cols:
        X_derived['PAY_STATUS_MEAN'] = X[pay_status_cols].mean(axis=1)
        X_derived['PAY_DELAY_COUNT'] = (X[pay_status_cols] > 0).sum(axis=1)
    
    # 计算信用额度利用率（如果有相关特征）
    if 'LIMIT_BAL' in X.columns and bill_cols:
        X_derived['CREDIT_UTILIZATION'] = X[bill_cols].mean(axis=1) / (X['LIMIT_BAL'] + 1e-8)
    
    logger.info(f"创建衍生特征，新增 {X_derived.shape[1] - X.shape[1]} 个特征")
    
    return X_derived


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    data = {
        'LIMIT_BAL': [20000, 120000, 90000],
        'SEX': [2, 2, 2],
        'EDUCATION': [2, 2, 2],
        'AGE': [24, 26, 34],
        'BILL_AMT1': [3913, 2682, 13559],
        'PAY_AMT1': [0, 0, 1518]
    }
    
    X = pd.DataFrame(data)
    
    # 测试特征工程
    fe = FeatureEngineer(scaler_type="standard")
    X_scaled = fe.fit_transform(X)
    
    print("原始数据:")
    print(X)
    print("\n缩放后数据:")
    print(X_scaled) 