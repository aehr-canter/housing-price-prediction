from loadandpreprocess import *
from trainandevaluate import *
from saveandload import *

def preprocess_new_data(X_new, fitted_preprocessor):
    return fitted_preprocessor.transform(X_new)


if __name__ == "__main__":
    X_processed, y, preprocessor = preprocess_housing_data(
        'AmesHousing.csv', 
        return_preprocessor=True
    )

    model, metrics, X_train, X_test, y_train, y_test = train_and_evaluate_model(X_processed, y)
    
    saved_files = save_model_and_preprocessor(model, preprocessor)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Model performance: {metrics}")

    loaded_model, loaded_preprocessor = load_model_and_preprocessor()

    from sklearn.ensemble import RandomForestRegressor
    rf_model, rf_metrics, _, _, _, _ = train_and_evaluate_model(
        X_processed, y, 
        model=RandomForestRegressor(random_state=42)
    )
    
    save_model_and_preprocessor(
        rf_model, preprocessor,
        model_filename='rf_house_price_model.pkl',
        preprocessor_filename='rf_preprocessor.pkl'
    )
