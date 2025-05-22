import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def preprocess_housing_data(csv_file_path, target_column='SalePrice', 
                           numeric_impute_strategy='median', 
                           categorical_impute_strategy='most_frequent',
                           return_preprocessor=False):
    
    # Load the dataset
    data = pd.read_csv(csv_file_path)
    
    # Separate the feature columns and the target column
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create pipelines for the numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_impute_strategy)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_impute_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine the pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Use the preprocessor on the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    if return_preprocessor:
        return X_preprocessed, y, preprocessor
    else:
        return X_preprocessed, y