"""数据处理模块测试."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer


class TestDataLoader:
    """数据加载器测试类."""
    
    def setup_method(self):
        """设置测试环境."""
        self.loader = DataLoader()
    
    def test_feature_names(self):
        """测试特征名称定义."""
        expected_features = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'AGE',
            'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3',
            'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]
        
        assert self.loader.feature_names == expected_features
        assert len(self.loader.feature_names) == 23
    
    @patch('src.data.data_loader.fetch_ucirepo')
    def test_load_taiwan_data(self, mock_fetch):
        """测试台湾数据加载."""
        # 模拟数据
        mock_data = MagicMock()
        mock_data.data.features = pd.DataFrame(
            np.random.randn(100, 23),
            columns=[f'X{i+1}' for i in range(23)]
        )
        mock_data.data.targets = pd.Series(np.random.randint(0, 2, 100))
        mock_fetch.return_value = mock_data
        
        X, y = self.loader.load_taiwan_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[1] == 23
        assert len(X) == len(y)
        assert list(X.columns) == self.loader.feature_names
    
    def test_calculate_feature_importance(self):
        """测试特征重要性计算."""
        # 创建测试数据
        X = pd.DataFrame(
            np.random.randn(100, 23),
            columns=self.loader.feature_names
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_selected, selected_features = self.loader.calculate_feature_importance(X, y, k=10)
        
        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selected_features, list)
        assert X_selected.shape[1] == 10
        assert len(selected_features) == 10
        assert self.loader.feature_importance is not None
    
    def test_split_data(self):
        """测试数据划分."""
        # 创建测试数据
        X = pd.DataFrame(
            np.random.randn(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_test, y_train, y_test = self.loader.split_data(X, y, test_size=0.3)
        
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(y_train) == 70
        assert len(y_test) == 30
        assert X_train.shape[1] == X_test.shape[1] == 10


class TestFeatureEngineer:
    """特征工程测试类."""
    
    def setup_method(self):
        """设置测试环境."""
        self.fe = FeatureEngineer(scaler_type="standard")
    
    def test_init(self):
        """测试初始化."""
        # 测试标准缩放器
        fe_std = FeatureEngineer(scaler_type="standard")
        assert fe_std.scaler_type == "standard"
        assert fe_std.scaler is not None
        
        # 测试最小最大缩放器
        fe_minmax = FeatureEngineer(scaler_type="minmax")
        assert fe_minmax.scaler_type == "minmax"
        
        # 测试无效类型
        with pytest.raises(ValueError):
            FeatureEngineer(scaler_type="invalid")
    
    def test_fit_transform(self):
        """测试拟合和转换."""
        # 创建测试数据
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [100, 200, 300, 400, 500]
        })
        
        # 拟合并转换
        X_transformed = self.fe.fit_transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)
        
        # 检查标准化后的数据均值接近0，标准差接近1
        assert abs(X_transformed.mean().mean()) < 0.1
        assert abs(X_transformed.std().mean() - 1) < 0.1
    
    def test_transform_without_fit(self):
        """测试未拟合就转换的情况."""
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            self.fe.transform(X)
    
    def test_get_feature_names(self):
        """测试获取特征名称."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        self.fe.fit(X)
        feature_names = self.fe.get_feature_names()
        
        assert feature_names == ['feature1', 'feature2']


if __name__ == "__main__":
    pytest.main([__file__]) 