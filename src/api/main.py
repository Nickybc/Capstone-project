"""FastAPI主应用."""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yaml
import uvicorn

from .schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    ExplanationResponse, HealthResponse, ErrorResponse,
    FeatureImportanceResponse
)
from ..models.predict import ModelPredictor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
predictor = None
app_config = None


def load_config():
    """加载配置文件."""
    global app_config
    try:
        with open("configs/api_config.yaml", 'r', encoding='utf-8') as f:
            app_config = yaml.safe_load(f)
        logger.info("配置文件加载成功")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        # 使用默认配置
        app_config = {
            "api": {
                "title": "Credit Card Default Prediction API",
                "description": "API for predicting credit card default probability",
                "version": "1.0.0"
            },
            "model_service": {
                "model_path": "models/stacking_model.pkl",
                "scaler_path": "models/scaler.pkl",
                "threshold": 0.5
            }
        }


def load_model():
    """加载预训练模型."""
    global predictor
    try:
        model_path = app_config["model_service"]["model_path"]
        scaler_path = app_config["model_service"]["scaler_path"]
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return False
        
        if not os.path.exists(scaler_path):
            logger.warning(f"缩放器文件不存在: {scaler_path}")
            return False
        
        predictor = ModelPredictor(model_path, scaler_path)
        logger.info("模型加载成功")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理."""
    # 启动时执行
    logger.info("启动信用卡违约预测API服务...")
    load_config()
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("模型未成功加载，部分功能可能不可用")
    
    yield
    
    # 关闭时执行
    logger.info("关闭API服务...")


# 创建FastAPI应用
app = FastAPI(
    title="Credit Card Default Prediction API",
    description="API for predicting credit card default probability using Stacking ensemble model",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor():
    """依赖注入：获取预测器实例."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请检查模型文件是否存在"
        )
    return predictor


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器."""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="服务器内部错误",
            details={"exception": str(exc)}
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径，返回API信息."""
    return {
        "message": "Credit Card Default Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_default(
    request: PredictionRequest,
    predictor_instance: ModelPredictor = Depends(get_predictor)
):
    """单个样本违约预测."""
    try:
        # 转换为字典格式
        features_dict = request.features.dict()
        
        # 进行预测
        results = predictor_instance.predict_with_confidence(
            [features_dict], 
            threshold=request.threshold
        )
        
        return PredictionResponse(**results[0])
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"预测失败: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor_instance: ModelPredictor = Depends(get_predictor)
):
    """批量违约预测."""
    try:
        # 转换为字典格式
        features_list = [features.dict() for features in request.features_list]
        
        # 进行批量预测
        results = predictor_instance.predict_with_confidence(
            features_list,
            threshold=request.threshold
        )
        
        # 统计高风险客户数量
        high_risk_count = sum(
            1 for result in results 
            if result['risk_level'] in ['高风险', '极高风险']
        )
        
        # 转换为响应格式
        predictions = [PredictionResponse(**result) for result in results]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            high_risk_count=high_risk_count
        )
        
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"批量预测失败: {str(e)}"
        )


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: PredictionRequest,
    predictor_instance: ModelPredictor = Depends(get_predictor)
):
    """预测解释，分析影响因素."""
    try:
        # 转换为字典格式
        features_dict = request.features.dict()
        
        # 获取预测解释
        explanation = predictor_instance.explain_prediction(features_dict)
        
        return ExplanationResponse(**explanation)
        
    except Exception as e:
        logger.error(f"预测解释失败: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"预测解释失败: {str(e)}"
        )


@app.get("/model/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    predictor_instance: ModelPredictor = Depends(get_predictor)
):
    """获取模型特征重要性."""
    try:
        # 获取特征重要性
        importance_dict = predictor_instance.get_feature_importance()
        
        if not importance_dict:
            raise HTTPException(
                status_code=404,
                detail="模型不支持特征重要性分析"
            )
        
        # 排序并获取前10个特征
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_features = [
            {"feature": feature, "importance": importance}
            for feature, importance in sorted_features
        ]
        
        return FeatureImportanceResponse(
            feature_importance=importance_dict,
            top_features=top_features
        )
        
    except Exception as e:
        logger.error(f"获取特征重要性失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取特征重要性失败: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """获取模型信息."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载"
        )
    
    try:
        return {
            "model_type": "Stacking Classifier",
            "base_learners": ["Random Forest", "Gradient Boosting", "XGBoost"],
            "final_estimator": "Logistic Regression",
            "feature_count": len(predictor.feature_engineer.get_feature_names()) if predictor.feature_engineer else 0,
            "model_path": app_config["model_service"]["model_path"],
            "scaler_path": app_config["model_service"]["scaler_path"]
        }
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取模型信息失败: {str(e)}"
        )


# 添加后台任务示例
@app.post("/predict/async")
async def predict_async(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    predictor_instance: ModelPredictor = Depends(get_predictor)
):
    """异步预测（示例）."""
    
    def log_prediction(features_dict: dict, result: dict):
        """记录预测结果到日志."""
        logger.info(f"异步预测完成: 输入={features_dict}, 结果={result}")
    
    try:
        # 转换为字典格式
        features_dict = request.features.dict()
        
        # 进行预测
        results = predictor_instance.predict_with_confidence(
            [features_dict],
            threshold=request.threshold
        )
        
        result = results[0]
        
        # 添加后台任务
        background_tasks.add_task(log_prediction, features_dict, result)
        
        return {
            "message": "预测已完成",
            "result": PredictionResponse(**result)
        }
        
    except Exception as e:
        logger.error(f"异步预测失败: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"异步预测失败: {str(e)}"
        )


if __name__ == "__main__":
    # 开发环境运行
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 