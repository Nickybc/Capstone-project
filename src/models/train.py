"""模型训练模块."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
import yaml
import joblib
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

import wandb
from ..data.data_loader import DataLoader
from ..data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器类，实现Stacking集成学习."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        初始化模型训练器.
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.feature_engineer = None
        self.training_history = {}
        
    def create_base_learners(self) -> list:
        """
        创建基础学习器.
        
        Returns:
            基础学习器列表
        """
        base_learners = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.config['data']['random_state'],
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config['data']['random_state']
            )),
            ('xgb', XGBClassifier(
                n_estimators=self.config['model']['params']['n_estimators'],
                max_depth=self.config['model']['params']['max_depth'],
                learning_rate=self.config['model']['params']['learning_rate'],
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=self.config['data']['random_state'],
                n_jobs=-1
            ))
        ]
        
        return base_learners
    
    def create_stacking_model(self) -> StackingClassifier:
        """
        创建Stacking集成模型.
        
        Returns:
            Stacking分类器
        """
        base_learners = self.create_base_learners()
        
        stacking_model = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(
                max_iter=1000,
                random_state=self.config['data']['random_state']
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
        """
        训练模型.
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            
        Returns:
            训练好的模型
        """
        logger.info("开始训练Stacking集成模型...")
        
        # 创建Stacking模型
        self.model = self.create_stacking_model()
        
        # 训练模型
        start_time = datetime.now()
        self.model.fit(X_train, y_train.values.ravel())
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
        
        # 记录训练历史
        self.training_history['training_time'] = training_time
        self.training_history['training_samples'] = len(X_train)
        self.training_history['features'] = X_train.columns.tolist()
        
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估模型性能.
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train_model 方法")
        
        logger.info("评估模型性能...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # 详细分类报告
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"模型评估结果:")
        logger.info(f"准确率: {metrics['accuracy']:.4f}")
        logger.info(f"F1分数: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # 保存评估结果
        self.training_history['test_metrics'] = metrics
        self.training_history['classification_report'] = classification_rep
        self.training_history['confusion_matrix'] = conf_matrix.tolist()
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        交叉验证.
        
        Args:
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
            
        Returns:
            交叉验证结果
        """
        if self.model is None:
            self.model = self.create_stacking_model()
        
        logger.info(f"执行 {cv} 折交叉验证...")
        
        # 交叉验证评分
        cv_scores = {
            'accuracy': cross_val_score(self.model, X, y, cv=cv, scoring='accuracy'),
            'f1': cross_val_score(self.model, X, y, cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        }
        
        # 计算统计信息
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
            cv_results[f'{metric}_scores'] = scores.tolist()
        
        logger.info("交叉验证结果:")
        for metric in ['accuracy', 'f1', 'roc_auc']:
            mean_score = cv_results[f'{metric}_mean']
            std_score = cv_results[f'{metric}_std']
            logger.info(f"{metric}: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        return cv_results
    
    def save_model(self, model_dir: str = "models") -> str:
        """
        保存训练好的模型.
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            模型保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train_model 方法")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = f"{model_dir}/stacking_model_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        
        # 保存训练历史
        history_path = f"{model_dir}/training_history_{timestamp}.pkl"
        joblib.dump(self.training_history, history_path)
        
        logger.info(f"模型已保存到: {model_path}")
        logger.info(f"训练历史已保存到: {history_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> StackingClassifier:
        """
        加载训练好的模型.
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        self.model = joblib.load(model_path)
        logger.info(f"模型已从 {model_path} 加载")
        
        return self.model
    
    def log_to_wandb(self, metrics: Dict[str, float], cv_results: Optional[Dict] = None) -> None:
        """
        将训练结果记录到Weights & Biases.
        
        Args:
            metrics: 评估指标
            cv_results: 交叉验证结果（可选）
        """
        if wandb.run is not None:
            # 记录基本指标
            wandb.log({
                "test/accuracy": metrics['accuracy'],
                "test/f1_score": metrics['f1_score'],
                "test/roc_auc": metrics['roc_auc'],
                "training/time_seconds": self.training_history.get('training_time', 0),
                "training/samples": self.training_history.get('training_samples', 0)
            })
            
            # 记录交叉验证结果
            if cv_results:
                for metric in ['accuracy', 'f1', 'roc_auc']:
                    wandb.log({
                        f"cv/{metric}_mean": cv_results[f'{metric}_mean'],
                        f"cv/{metric}_std": cv_results[f'{metric}_std']
                    })
            
            # 记录模型配置
            wandb.config.update(self.config)
    
    def train_pipeline(self, use_wandb: bool = True) -> Tuple[StackingClassifier, Dict[str, float]]:
        """
        完整的训练流水线.
        
        Args:
            use_wandb: 是否使用Weights & Biases记录
            
        Returns:
            训练好的模型和评估指标
        """
        # 初始化wandb
        if use_wandb:
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb'].get('entity'),
                tags=self.config['wandb']['tags'],
                notes=self.config['wandb']['notes'],
                config=self.config
            )
        
        try:
            # 加载数据
            data_loader = DataLoader()
            X_train, X_test, y_train, y_test = data_loader.process_pipeline()
            
            # 特征工程
            self.feature_engineer = FeatureEngineer(scaler_type="standard")
            X_train_scaled = self.feature_engineer.fit_transform(X_train)
            X_test_scaled = self.feature_engineer.transform(X_test)
            
            # 保存特征工程器
            self.feature_engineer.save_scaler("models/scaler.pkl")
            
            # 训练模型
            self.train_model(X_train_scaled, y_train)
            
            # 交叉验证
            cv_results = self.cross_validate(X_train_scaled, y_train)
            
            # 评估模型
            metrics = self.evaluate_model(X_test_scaled, y_test)
            
            # 保存模型
            model_path = self.save_model()
            
            # 记录到wandb
            if use_wandb:
                self.log_to_wandb(metrics, cv_results)
                
                # 保存模型到wandb artifacts
                model_artifact = wandb.Artifact(
                    name="stacking-model",
                    type="model",
                    description="Stacking ensemble model for credit card default prediction"
                )
                model_artifact.add_file(model_path)
                model_artifact.add_file("models/scaler.pkl")
                wandb.log_artifact(model_artifact)
            
            return self.model, metrics
            
        finally:
            if use_wandb and wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建训练器并运行训练流水线
    trainer = ModelTrainer()
    model, metrics = trainer.train_pipeline()
    
    print(f"训练完成！")
    print(f"测试集性能: {metrics}") 