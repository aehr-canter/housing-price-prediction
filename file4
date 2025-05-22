import joblib

def save_model_and_preprocessor(model, preprocessor, model_filename='house_price_model.pkl', 
                              preprocessor_filename='preprocessor.pkl', verbose=True):
    # Save the model
    joblib.dump(model, model_filename)
    
    # Save the preprocessor
    joblib.dump(preprocessor, preprocessor_filename)
    
    if verbose:
        print(f'Model saved as: {model_filename}')
        print(f'Preprocessor saved as: {preprocessor_filename}')
    
    return {
        'model_file': model_filename,
        'preprocessor_file': preprocessor_filename
    }


def load_model_and_preprocessor(model_filename='house_price_model.pkl', 
                              preprocessor_filename='preprocessor.pkl'):

    model = joblib.load(model_filename)
    preprocessor = joblib.load(preprocessor_filename)
    
    return model, preprocessor