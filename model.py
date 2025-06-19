from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def logistic_regression_model():
    """Return configured Logistic Regression model"""
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    return model

def random_forest_model():
    """Return configured Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    return model

def xgboost_model():
    """Return configured XGBoost model"""
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        gamma=0,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=1,
        n_jobs=-1,
        eval_metric='logloss'
    )
    return model

def lightgbm_model():
    """Return configured LightGBM model"""
    model = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=200,
        subsample_for_bin=200000,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=0,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        importance_type='split'
    )
    return model
