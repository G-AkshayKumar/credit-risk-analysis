from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from model_definitions import *

def create_preprocessor(X):
    """Create preprocessing pipeline"""
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    """Train and tune multiple models"""
    print("\n=== Model Training ===")
    
    # Define models
    models = {
        'Logistic Regression': logistic_regression_model(),
        'Random Forest': random_forest_model(),
        'XGBoost': xgboost_model(),
        'LightGBM': lightgbm_model()
    }
    
    # Hyperparameter grids for tuning
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2']
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 6, 9]
        },
        'LightGBM': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__num_leaves': [31, 63, 127]
        }
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1)
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        trained_models[name] = grid_search.best_estimator_
        print(f"Best params for {name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Save model
        joblib.dump(grid_search.best_estimator_, f'{name.lower().replace(" ", "_")}_model.pkl')
    
    return trained_models

def shap_analysis(model, X_train, preprocessor, model_name):
    """Perform SHAP analysis for model interpretability"""
    print(f"\n=== SHAP Analysis for {model_name} ===")
    
    # Process the data through the pipeline steps except the final estimator
    processed_data = preprocessor.transform(X_train)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    # For tree-based models
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(processed_data)
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], processed_data, feature_names=feature_names)
        else:
            shap.summary_plot(shap_values, processed_data, feature_names=feature_names)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{model_name.lower().replace(" ", "_")}.png')
        plt.show()
    
    # For linear models
    elif hasattr(model.named_steps['classifier'], 'coef_'):
        explainer = shap.LinearExplainer(model.named_steps['classifier'], processed_data)
        shap_values = explainer.shap_values(processed_data)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, processed_data, feature_names=feature_names)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{model_name.lower().replace(" ", "_")}.png')
        plt.show()

def get_feature_names(preprocessor):
    """Get feature names after preprocessing"""
    # Get numerical feature names
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    
    # Get categorical feature names
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        preprocessor.named_transformers_['cat'].feature_names_in_)
    
    # Combine all feature names
    all_features = np.concatenate([num_features, cat_features])
    return all_features

