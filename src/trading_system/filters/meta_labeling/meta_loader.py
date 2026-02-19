import importlib
import logging
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import xgboost as xgb

from src.trading_system.config.meta_configs import META_LABELING_CONFIG

logger = logging.getLogger(__name__)


def _get_config(strategy_name: str, key: str = 'enabled') -> dict | None:
    config = META_LABELING_CONFIG.get(strategy_name)
    if not config or not config.get(key, False):
        return None
    return config


def load_meta_model(strategy_name: str) -> Any | None:
    if not _get_config(strategy_name):
        return None

    model_path = Path(f"strategies/{strategy_name}/models/prod_v1.json")
    if not model_path.exists():
        logger.warning(f"{strategy_name}: Meta model not found | path={model_path}")
        return None

    try:
        import xgboost as xgb
    except ImportError:
        logger.warning(f"{strategy_name}: xgboost not installed; meta-labeling disabled")
        return None

    try:
        model = xgb.XGBClassifier()
        model._estimator_type = "classifier"
        model.load_model(model_path.as_posix())
        return model
    except Exception as error:
        logger.warning(f"{strategy_name}: Failed to load meta model | err={error}")
        return None


def load_calibration_model(strategy_name: str) -> Any | None:
    if not _get_config(strategy_name, 'use_calibration'):
        return None

    calibration_dir = Path(f"strategies/{strategy_name}/calibration_models")
    if not calibration_dir.exists():
        logger.warning(f"{strategy_name}: Calibration directory not found | path={calibration_dir}")
        return None

    from src.trading_system.filters.meta_labeling.probability_calibrator.calibrator import ProbabilityCalibrator

    try:
        return ProbabilityCalibrator.load_state(calibration_dir.as_posix())
    except Exception as error:
        logger.warning(f"{strategy_name}: Failed to load calibration model | err={error}")
        return None


def load_features_extractor(strategy_name: str) -> Callable | None:
    if not _get_config(strategy_name):
        return None

    try:
        module = importlib.import_module(f"strategies.{strategy_name}.features")
    except Exception as error:
        logger.warning(f"{strategy_name}: Failed to import features module | err={error}")
        return None

    extractor = getattr(module, "extract_features", None)
    if extractor is None or not callable(extractor):
        logger.error(f"{strategy_name}: features.py exists but no extract_features function | fn=missing")
        return None

    return extractor


def get_min_confidence(strategy_name: str) -> float:
    config = META_LABELING_CONFIG.get(strategy_name)
    if not config:
        return 0.0
    try:
        return float(config.get('min_confidence', 0.0))
    except (TypeError, ValueError):
        logger.warning(f"{strategy_name}: Invalid min_confidence in config | fallback=0.0")
        return 0.0
