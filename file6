from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model_performance(y_true, y_pred, print_results=True):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    if print_results:
        print(f'MAE: {mae:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'R²: {r2:.4f}')
    
    return metrics