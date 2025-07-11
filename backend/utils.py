# NEW utils.py

import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import tempfile
from pathlib import Path

class MaterialAnalysisError(Exception):
    """Custom exception for material analysis errors"""
    pass

class DataLoadError(MaterialAnalysisError):
    """Error loading or processing data"""
    pass

class PredictionError(MaterialAnalysisError):
    """Error in prediction calculations"""
    pass

def setup_web_logging(name: str = 'material_analyzer') -> logging.Logger:
    """Setup logging for web application"""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def serialize_for_json(obj: Any) -> Any:
    """Convert pandas/numpy objects to JSON-serializable format"""
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif pd.isna(obj):
        return None
    return obj

def clean_json_response(data: Dict) -> Dict:
    """Clean response data for JSON serialization"""
    if isinstance(data, dict):
        return {k: serialize_for_json(v) if not isinstance(v, dict) 
                else clean_json_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_response(item) if isinstance(item, dict) 
                else serialize_for_json(item) for item in data]
    return serialize_for_json(data)

def handle_exceptions(func):
    """Decorator to handle exceptions in web routes"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataLoadError as e:
            logger = setup_web_logging()
            logger.error(f"Data load error: {str(e)}")
            return {
                'success': False,
                'error': 'Data loading failed',
                'message': str(e)
            }, 400
        except PredictionError as e:
            logger = setup_web_logging()
            logger.error(f"Prediction error: {str(e)}")
            return {
                'success': False,
                'error': 'Prediction failed',
                'message': str(e)
            }, 500
        except MaterialAnalysisError as e:
            logger = setup_web_logging()
            logger.error(f"Material analysis error: {str(e)}")
            return {
                'success': False,
                'error': 'Analysis failed',
                'message': str(e)
            }, 500
        except Exception as e:
            logger = setup_web_logging()
            logger.error(f"Unexpected error: {str(e)}")
            return {
                'success': False,
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }, 500
    return wrapper

def create_temp_file(suffix: str = '.xlsx') -> str:
    """Create temporary file for processing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name

def cleanup_temp_file(file_path: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass

def validate_file_size(file_path: str, max_size_mb: int = 50) -> bool:
    """Validate file size"""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def get_month_names() -> Dict[int, str]:
    """Get month number to name mapping"""
    return {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

class ProgressTracker:
    """Track processing progress for web interface"""
    
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        self.status = "initialized"
        self.message = ""
    
    def start(self, total_steps: int, message: str = "Starting..."):
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "running"
        self.message = message
    
    def update(self, step: int, message: str = ""):
        self.current_step = step
        self.message = message
        if step >= self.total_steps:
            self.status = "completed"
    
    def get_progress(self) -> Dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percentage": int((self.current_step / self.total_steps) * 100) if self.total_steps > 0 else 0,
            "status": self.status,
            "message": self.message
        }