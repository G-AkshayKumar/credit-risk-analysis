import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the dataset"""
    df = pd.read_csv('data-3.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic info
    print("\nData Info:")
    print(df.info())
    
    # Missing values
    print("\nMissing Values Summary:")
    print(df.isnull().sum())
    
    # Numerical features summary
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        print("\nNumerical Features Summary:")
        print(df[num_cols].describe())
        
        # Plot distributions
        df[num_cols].hist(bins=20, figsize=(15, 10))
        plt.suptitle('Numerical Features Distribution')
        plt.show()
    
    # Categorical features summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print("\nCategorical Features Summary:")
        for col in cat_cols:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())
            
            # Plot value counts
            plt.figure(figsize=(10, 5))
            sns.countplot(x=col, data=df)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)
            plt.show()
    
    # Correlation matrix
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()
    
    return df

def feature_engineering(df):
    """Create new features and transform existing ones"""
    print("\n=== Feature Engineering ===")
    
    # Example features (adjust based on your actual data)
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                labels=['<30', '30-40', '40-50', '50-60', '60+'])
    
    if 'income' in df.columns and 'debt' in df.columns:
        df['debt_to_income'] = df['debt'] / (df['income'] + 1e-6)
        df['income_debt_diff'] = df['income'] - df['debt']
    
    if 'credit_score' in df.columns:
        df['credit_score_group'] = pd.cut(df['credit_score'], 
                                         bins=[0, 500, 600, 700, 850],
                                         labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    if 'payment_history' in df.columns:
        df['late_payments'] = df['payment_history'].apply(lambda x: x.count('late'))
    
    # Date features example
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
        except:
            continue
    
    print(f"Added {len([col for col in df.columns if col not in original_cols])} new features")
    return df

def preprocess_data(df, target_col='default'):
    """Prepare data for modeling"""
    print("\n=== Data Preprocessing ===")
    
    # Handle missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Simple imputation
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y
