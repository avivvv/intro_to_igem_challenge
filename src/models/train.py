"""Training utilities for expression prediction models."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib


def create_baseline_models():
    """
    Create a dictionary of baseline models.
    
    Returns:
        Dictionary of scikit-learn models
    """
    models = {
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    return models


def train_model(X, y, model, test_size=0.2, random_state=42):
    """
    Train and evaluate a regression model.
    
    Args:
        X: Feature matrix
        y: Target values
        model: Scikit-learn model
        test_size: Test set fraction
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, scaler, train_score, test_score)
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, scaler, train_score, test_score


def save_model(model, scaler, model_path, scaler_path):
    """
    Save trained model and scaler.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_path: Path to save model
        scaler_path: Path to save scaler
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_path, scaler_path):
    """
    Load trained model and scaler.
    
    Args:
        model_path: Path to model file
        scaler_path: Path to scaler file
        
    Returns:
        Tuple of (model, scaler)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
