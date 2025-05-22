import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_evaluate_model(X_preprocessed, y, model=None, test_size=0.2, 
                           random_state=42, save_model=False, model_filename=None):
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize model if not provided
    if model is None:
        model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Store metrics in dictionary
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    # Print metrics
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R²: {r2:.4f}')
    
    if save_model:
        filename = model_filename or 'trained_model.pkl'
        joblib.dump(model, filename)
        print(f'Model saved as {filename}')
    
    return model, metrics, X_train, X_test, y_train, y_test