"""模型预测模块."""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Tuple
import logging
import joblib
import yaml
from pathlib import Path

from ..data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelPredictor:
    """模型预测器类，用于加载模型并进行预测."""
    
    def __init__(self, model_path: str, scaler_path: str, 
                 config_path: str = "configs/model_config.yaml"):
        """
        初始化模型预测器.
        
        Args:
            model_path: 模型文件路径
            scaler_path: 缩放器文件路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.feature_engineer = None
        self.feature_names = None
        
        # 加载模型和预处理器
        self._load_model()
        self._load_preprocessor()
    
    def _load_model(self) -> None:
        """加载训练好的模型."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"模型已从 {self.model_path} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _load_preprocessor(self) -> None:
        """加载特征预处理器."""
        try:
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.load_scaler(self.scaler_path)
            logger.info(f"预处理器已从 {self.scaler_path} 加载")
        except Exception as e:
            logger.error(f"加载预处理器失败: {e}")
            raise
    
    def preprocess_input(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        预处理输入数据.
        
        Args:
            input_data: 输入数据，可以是DataFrame、字典或字典列表
            
        Returns:
            预处理后的DataFrame
        """
        # 转换为DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("输入数据必须是DataFrame、字典或字典列表")
        
        # 验证必要特征
        expected_features = self.feature_engineer.get_feature_names()
        if expected_features:
            missing_features = set(expected_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"缺少必要特征: {missing_features}")
            
            # 确保特征顺序一致
            df = df[expected_features]
        
        # 应用特征工程
        df_scaled = self.feature_engineer.transform(df)
        
        return df_scaled
    
    def predict(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        进行预测.
        
        Args:
            input_data: 输入数据
            
        Returns:
            预测结果（0或1）
        """
        # 预处理输入数据
        X_processed = self.preprocess_input(input_data)
        
        # 进行预测
        predictions = self.model.predict(X_processed)
        
        logger.info(f"完成预测，样本数: {len(predictions)}")
        
        return predictions
    
    def predict_proba(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        预测概率.
        
        Args:
            input_data: 输入数据
            
        Returns:
            预测概率，形状为 (n_samples, 2)
        """
        # 预处理输入数据
        X_processed = self.preprocess_input(input_data)
        
        # 预测概率
        probabilities = self.model.predict_proba(X_processed)
        
        logger.info(f"完成概率预测，样本数: {len(probabilities)}")
        
        return probabilities
    
    def predict_default_probability(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        预测违约概率（正类概率）.
        
        Args:
            input_data: 输入数据
            
        Returns:
            违约概率
        """
        probabilities = self.predict_proba(input_data)
        return probabilities[:, 1]  # 返回正类（违约）概率
    
    def predict_with_confidence(self, input_data: Union[pd.DataFrame, Dict, List[Dict]], 
                              threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        带置信度的预测.
        
        Args:
            input_data: 输入数据
            threshold: 分类阈值
            
        Returns:
            包含预测结果和置信度的字典列表
        """
        # 获取概率预测
        probabilities = self.predict_proba(input_data)
        default_probs = probabilities[:, 1]
        
        # 生成预测结果
        predictions = (default_probs >= threshold).astype(int)
        
        # 计算置信度（最大概率）
        confidence = np.max(probabilities, axis=1)
        
        results = []
        for i, (pred, prob, conf) in enumerate(zip(predictions, default_probs, confidence)):
            result = {
                'prediction': int(pred),
                'default_probability': float(prob),
                'confidence': float(conf),
                'risk_level': self._get_risk_level(prob)
            }
            results.append(result)
        
        return results
    
    def _get_risk_level(self, probability: float) -> str:
        """
        根据违约概率确定风险等级.
        
        Args:
            probability: 违约概率
            
        Returns:
            风险等级字符串
        """
        if probability < 0.2:
            return "低风险"
        elif probability < 0.5:
            return "中等风险"
        elif probability < 0.8:
            return "高风险"
        else:
            return "极高风险"
    
    def batch_predict(self, data_path: str, output_path: str = None) -> pd.DataFrame:
        """
        批量预测.
        
        Args:
            data_path: 输入数据文件路径
            output_path: 输出文件路径（可选）
            
        Returns:
            包含预测结果的DataFrame
        """
        # 读取数据
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            raise ValueError("支持的文件格式: .csv, .xlsx")
        
        logger.info(f"从 {data_path} 读取了 {len(data)} 个样本")
        
        # 进行预测
        predictions = self.predict(data)
        probabilities = self.predict_default_probability(data)
        
        # 创建结果DataFrame
        results_df = data.copy()
        results_df['prediction'] = predictions
        results_df['default_probability'] = probabilities
        results_df['risk_level'] = [self._get_risk_level(p) for p in probabilities]
        
        # 保存结果
        if output_path:
            if output_path.endswith('.csv'):
                results_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                results_df.to_excel(output_path, index=False)
            logger.info(f"预测结果已保存到 {output_path}")
        
        return results_df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（如果模型支持）.
        
        Returns:
            特征重要性字典
        """
        importance_dict = {}
        
        # 对于Stacking模型，获取基础学习器的特征重要性
        if hasattr(self.model, 'estimators_'):
            for name, estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    feature_names = self.feature_engineer.get_feature_names()
                    importances = estimator.feature_importances_
                    
                    for feature, importance in zip(feature_names, importances):
                        if feature not in importance_dict:
                            importance_dict[feature] = []
                        importance_dict[feature].append(importance)
            
            # 计算平均重要性
            avg_importance = {
                feature: np.mean(values) 
                for feature, values in importance_dict.items()
            }
            
            return avg_importance
        
        return {}
    
    def explain_prediction(self, input_data: Dict) -> Dict[str, Any]:
        """
        解释单个预测结果.
        
        Args:
            input_data: 单个样本数据
            
        Returns:
            预测解释
        """
        # 获取预测结果
        prediction_result = self.predict_with_confidence([input_data])[0]
        
        # 获取特征重要性
        feature_importance = self.get_feature_importance()
        
        # 分析重要特征值
        important_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # 取前5个重要特征
        
        feature_analysis = []
        for feature, importance in important_features:
            if feature in input_data:
                feature_analysis.append({
                    'feature': feature,
                    'value': input_data[feature],
                    'importance': importance
                })
        
        explanation = {
            'prediction': prediction_result,
            'top_features': feature_analysis,
            'explanation': self._generate_explanation(prediction_result, feature_analysis)
        }
        
        return explanation
    
    def _generate_explanation(self, prediction_result: Dict, feature_analysis: List[Dict]) -> str:
        """
        生成预测解释文本.
        
        Args:
            prediction_result: 预测结果
            feature_analysis: 特征分析
            
        Returns:
            解释文本
        """
        risk_level = prediction_result['risk_level']
        probability = prediction_result['default_probability']
        
        explanation = f"该客户被评估为{risk_level}，违约概率为 {probability:.2%}。"
        
        if feature_analysis:
            explanation += " 主要影响因素包括："
            for feature_info in feature_analysis[:3]:  # 只解释前3个特征
                feature = feature_info['feature']
                value = feature_info['value']
                explanation += f" {feature}({value})，"
            explanation = explanation.rstrip("，") + "。"
        
        return explanation


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 示例用法
    try:
        # 假设模型文件存在
        predictor = ModelPredictor(
            model_path="models/stacking_model.pkl",
            scaler_path="models/scaler.pkl"
        )
        
        # 示例输入数据
        sample_data = {
            'LIMIT_BAL': 20000,
            'SEX': 2,
            'EDUCATION': 2,
            'AGE': 24,
            'MARRIAGE': 1,
            'PAY_0': 2,
            'PAY_2': 2,
            'PAY_3': -1,
            'PAY_4': -1,
            'PAY_5': -2,
            'PAY_6': -2,
            'BILL_AMT1': 3913,
            'BILL_AMT2': 3102,
            'BILL_AMT3': 689,
            'BILL_AMT4': 0,
            'BILL_AMT5': 0,
            'BILL_AMT6': 0,
            'PAY_AMT1': 0,
            'PAY_AMT2': 689,
            'PAY_AMT3': 0,
            'PAY_AMT4': 0,
            'PAY_AMT5': 0,
            'PAY_AMT6': 0
        }
        
        # 进行预测
        result = predictor.predict_with_confidence(sample_data)
        print(f"预测结果: {result}")
        
        # 解释预测
        explanation = predictor.explain_prediction(sample_data)
        print(f"预测解释: {explanation}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保模型文件存在") 