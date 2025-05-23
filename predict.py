import joblib
from saveandload import *

def load_and_predict(model_filename, X_new):

    model = joblib.load(model_filename)
    predictions = model.predict(X_new)
    return predictions


def predict_on_new_data(new_data, model_filename='house_price_model.pkl', 
                       preprocessor_filename='preprocessor.pkl'):
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(model_filename, preprocessor_filename)
    
    # Preprocess the new data
    X_new_processed = preprocessor.transform(new_data)
    
    # Make predictions
    predictions = model.predict(X_new_processed)
    
    return predictions
