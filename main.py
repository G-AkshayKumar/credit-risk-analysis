from EDA import *
from model-training import *
import pandas as pd

def main():
    # Load data
    data = load_data('credit_data.csv')
    original_cols = data.columns.tolist()
    
    # Perform EDA
    data = perform_eda(data)
    
    # Feature engineering
    data = feature_engineering(data)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # Train models
    trained_models = train_models(X_train, y_train, preprocessor)
    
    # SHAP analysis for each model
    for name, model in trained_models.items():
        shap_analysis(model, X_train.sample(1000, random_state=42), preprocessor, name)
    
    # Save results
    pd.DataFrame(results).to_csv('model_results.csv')

if __name__ == "__main__":
    main()
