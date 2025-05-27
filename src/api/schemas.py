"""API数据模型和验证schemas."""

from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    """风险等级枚举."""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    VERY_HIGH = "极高风险"


class CreditCardFeatures(BaseModel):
    """信用卡特征数据模型."""
    
    # 基本信息
    LIMIT_BAL: float = Field(..., ge=0, description="信用额度（新台币）")
    SEX: int = Field(..., ge=1, le=2, description="性别 (1=男性, 2=女性)")
    EDUCATION: int = Field(..., ge=1, le=4, description="教育程度 (1=研究生, 2=大学, 3=高中, 4=其他)")
    MARRIAGE: int = Field(..., ge=1, le=3, description="婚姻状况 (1=已婚, 2=单身, 3=其他)")
    AGE: int = Field(..., ge=18, le=100, description="年龄")
    
    # 还款状态 (过去6个月)
    PAY_0: int = Field(..., ge=-2, le=9, description="9月还款状态")
    PAY_2: int = Field(..., ge=-2, le=9, description="8月还款状态")
    PAY_3: int = Field(..., ge=-2, le=9, description="7月还款状态")
    PAY_4: int = Field(..., ge=-2, le=9, description="6月还款状态")
    PAY_5: int = Field(..., ge=-2, le=9, description="5月还款状态")
    PAY_6: int = Field(..., ge=-2, le=9, description="4月还款状态")
    
    # 账单金额 (过去6个月)
    BILL_AMT1: float = Field(..., description="9月账单金额")
    BILL_AMT2: float = Field(..., description="8月账单金额")
    BILL_AMT3: float = Field(..., description="7月账单金额")
    BILL_AMT4: float = Field(..., description="6月账单金额")
    BILL_AMT5: float = Field(..., description="5月账单金额")
    BILL_AMT6: float = Field(..., description="4月账单金额")
    
    # 支付金额 (过去6个月)
    PAY_AMT1: float = Field(..., ge=0, description="9月支付金额")
    PAY_AMT2: float = Field(..., ge=0, description="8月支付金额")
    PAY_AMT3: float = Field(..., ge=0, description="7月支付金额")
    PAY_AMT4: float = Field(..., ge=0, description="6月支付金额")
    PAY_AMT5: float = Field(..., ge=0, description="5月支付金额")
    PAY_AMT6: float = Field(..., ge=0, description="4月支付金额")
    
    @validator('LIMIT_BAL')
    def validate_credit_limit(cls, v):
        """验证信用额度."""
        if v <= 0:
            raise ValueError('信用额度必须大于0')
        if v > 10000000:  # 1000万台币
            raise ValueError('信用额度超出合理范围')
        return v
    
    @validator('PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6')
    def validate_payment_status(cls, v):
        """验证还款状态."""
        valid_values = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if v not in valid_values:
            raise ValueError(f'还款状态必须在{valid_values}中')
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionRequest(BaseModel):
    """单个预测请求模型."""
    
    features: CreditCardFeatures = Field(..., description="信用卡特征数据")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="分类阈值")
    
    class Config:
        schema_extra = {
            "example": {
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
                },
                "threshold": 0.5
            }
        }


class BatchPredictionRequest(BaseModel):
    """批量预测请求模型."""
    
    features_list: List[CreditCardFeatures] = Field(..., description="特征数据列表")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="分类阈值")
    
    @validator('features_list')
    def validate_features_list(cls, v):
        """验证特征列表."""
        if len(v) == 0:
            raise ValueError('特征列表不能为空')
        if len(v) > 1000:
            raise ValueError('单次批量预测最多支持1000个样本')
        return v


class PredictionResponse(BaseModel):
    """预测响应模型."""
    
    prediction: int = Field(..., description="预测结果 (0=不违约, 1=违约)")
    default_probability: float = Field(..., ge=0.0, le=1.0, description="违约概率")
    confidence: float = Field(..., ge=0.0, le=1.0, description="预测置信度")
    risk_level: RiskLevel = Field(..., description="风险等级")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "default_probability": 0.75,
                "confidence": 0.82,
                "risk_level": "高风险"
            }
        }


class BatchPredictionResponse(BaseModel):
    """批量预测响应模型."""
    
    predictions: List[PredictionResponse] = Field(..., description="预测结果列表")
    total_count: int = Field(..., description="总预测数量")
    high_risk_count: int = Field(..., description="高风险客户数量")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "default_probability": 0.75,
                        "confidence": 0.82,
                        "risk_level": "高风险"
                    }
                ],
                "total_count": 1,
                "high_risk_count": 1
            }
        }


class ExplanationResponse(BaseModel):
    """预测解释响应模型."""
    
    prediction: PredictionResponse = Field(..., description="预测结果")
    top_features: List[Dict[str, Any]] = Field(..., description="重要特征分析")
    explanation: str = Field(..., description="解释文本")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": {
                    "prediction": 1,
                    "default_probability": 0.75,
                    "confidence": 0.82,
                    "risk_level": "高风险"
                },
                "top_features": [
                    {
                        "feature": "PAY_0",
                        "value": 2,
                        "importance": 0.15
                    }
                ],
                "explanation": "该客户被评估为高风险，违约概率为 75.00%。主要影响因素包括 PAY_0(2)。"
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应模型."""
    
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    version: str = Field(..., description="API版本")
    timestamp: str = Field(..., description="检查时间戳")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """错误响应模型."""
    
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "输入数据验证失败",
                "details": {
                    "field": "LIMIT_BAL",
                    "issue": "信用额度必须大于0"
                }
            }
        }


# 特征重要性响应模型
class FeatureImportanceResponse(BaseModel):
    """特征重要性响应模型."""
    
    feature_importance: Dict[str, float] = Field(..., description="特征重要性字典")
    top_features: List[Dict[str, Any]] = Field(..., description="前10重要特征")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_importance": {
                    "PAY_0": 0.15,
                    "PAY_2": 0.12,
                    "LIMIT_BAL": 0.10
                },
                "top_features": [
                    {"feature": "PAY_0", "importance": 0.15},
                    {"feature": "PAY_2", "importance": 0.12}
                ]
            }
        } 