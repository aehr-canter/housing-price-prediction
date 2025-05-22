import streamlit as st
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any, Optional


def load_model_components(model_path: str = 'house_price_model.pkl', 
                         preprocessor_path: str = 'preprocessor.pkl'):
    
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def get_default_features() -> Dict[str, Any]:

    return {
        'Central Air': 'Y',
        'House Style': '1Story',
        'Bsmt Full Bath': 1,
        'Kitchen Qual': 'TA',
        'Bsmt Qual': 'TA',
        'Exter Cond': 'TA',
        'Neighborhood': 'NAmes',
        'Garage Qual': 'TA',
        'Bedroom AbvGr': 3,
        'MS Zoning': 'RL',
        'Foundation': 'PConc',
        'Misc Val': 0,
        'Fireplaces': 0,
        'Bsmt Unf SF': 400,
        'Low Qual Fin SF': 0,
        'Land Contour': 'Lvl',
        'Bldg Type': '1Fam',
        'Garage Cond': 'TA',
        'Bsmt Cond': 'TA',
        'Alley': 'NA',
        'Condition 2': 'Norm',
        'Condition 1': 'Norm',
        'Wood Deck SF': 0,
        'Overall Cond': 5,
        'Lot Config': 'Inside',
        'Screen Porch': 0,
        'Lot Shape': 'Reg',
        '3Ssn Porch': 0,
        'Fence': 'NA',
        'Exterior 1st': 'VinylSd',
        'Land Slope': 'Gtl',
        'Heating QC': 'Ex',
        'Street': 'Pave',
        'Utilities': 'AllPub',
        'MS SubClass': 20,
        'Exterior 2nd': 'VinylSd',
        'Roof Style': 'Gable',
        'Open Porch SF': 20,
        'Bsmt Exposure': 'No',
        'Kitchen AbvGr': 1,
        'Paved Drive': 'Y',
        'Year Remod/Add': 2000,
        'Garage Area': 500,
        'Pool QC': 'NA',
        'Electrical': 'SBrkr',
        'Roof Matl': 'CompShg',
        'Sale Condition': 'Normal',
        'Mo Sold': 6,
        'Misc Feature': 'NA',
        'Bsmt Half Bath': 0,
        'Sale Type': 'WD',
        'Half Bath': 1,
        'Garage Type': 'Attchd',
        'Heating': 'GasA',
        'BsmtFin SF 1': 500,
        'Yr Sold': 2010,
        'Functional': 'Typ',
        'Pool Area': 0,
        'Exter Qual': 'TA',
        'Garage Finish': 'Unf',
        'Mas Vnr Type': 'None',
        'Mas Vnr Area': 0,
        'Garage Yr Blt': 1990,
        'Enclosed Porch': 0,
        'TotRms AbvGrd': 6,
        'Order': 1,
        'Fireplace Qu': 'NA',
        'BsmtFin Type 2': 'NA',
        'PID': 0,
        'BsmtFin Type 1': 'GLQ',
        'Lot Frontage': 60,
        'BsmtFin SF 2': 0,
        '2nd Flr SF': 0,
    }


def create_user_inputs() -> Dict[str, Any]:

    st.subheader("Key House Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lot_area = st.number_input('Lot Area (sq ft)', min_value=0, value=5000)
        overall_qual = st.number_input('Overall Quality', min_value=1, max_value=10, value=5)
        year_built = st.number_input('Year Built', min_value=1800, max_value=2024, value=1990)
        total_bsmt_sf = st.number_input('Total Basement SF', min_value=0, value=1000)
    
    with col2:
        first_flr_sf = st.number_input('1st Floor SF', min_value=0, value=1000)
        full_bath = st.number_input('Full Bathrooms', min_value=0, value=2)
        gr_liv_area = st.number_input('Above Ground Living Area', min_value=0, value=1500)
        garage_cars = st.number_input('Garage Cars', min_value=0, value=1)
    
    return {
        'Lot Area': lot_area,
        'Overall Qual': overall_qual,
        'Year Built': year_built,
        'Total Bsmt SF': total_bsmt_sf,
        '1st Flr SF': first_flr_sf,
        'Full Bath': full_bath,
        'Gr Liv Area': gr_liv_area,
        'Garage Cars': garage_cars,
    }


def create_advanced_inputs() -> Dict[str, Any]:

    advanced_inputs = {}
    
    if st.checkbox("Show Advanced Options"):
        with st.expander("Advanced House Features"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                advanced_inputs['Neighborhood'] = st.selectbox(
                    'Neighborhood', 
                    ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW'],
                    index=0
                )
                advanced_inputs['House Style'] = st.selectbox(
                    'House Style',
                    ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer'],
                    index=0
                )
                advanced_inputs['Overall Cond'] = st.number_input('Overall Condition', min_value=1, max_value=10, value=5)
            
            with col2:
                advanced_inputs['Kitchen Qual'] = st.selectbox('Kitchen Quality', ['TA', 'Gd', 'Ex', 'Fa'], index=0)
                advanced_inputs['Exter Qual'] = st.selectbox('Exterior Quality', ['TA', 'Gd', 'Ex', 'Fa'], index=0)
                advanced_inputs['Fireplaces'] = st.number_input('Fireplaces', min_value=0, value=0)
            
            with col3:
                advanced_inputs['Garage Area'] = st.number_input('Garage Area', min_value=0, value=500)
                advanced_inputs['Wood Deck SF'] = st.number_input('Wood Deck SF', min_value=0, value=0)
                advanced_inputs['2nd Flr SF'] = st.number_input('2nd Floor SF', min_value=0, value=0)
    
    return advanced_inputs


def create_input_dataframe(user_inputs: Dict[str, Any], advanced_inputs: Dict[str, Any]) -> pd.DataFrame:

    features = get_default_features()
    
    features.update(user_inputs)
    features.update(advanced_inputs)
    
    input_data = pd.DataFrame({key: [value] for key, value in features.items()})
    
    return input_data


def make_prediction(model, preprocessor, input_data: pd.DataFrame) -> float:

    try:

        input_features_preprocessed = preprocessor.transform(input_data)
        
        prediction = model.predict(input_features_preprocessed)
        
        return prediction[0]
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return 0.0


def display_prediction_result(prediction: float):

    st.success(f"🏠 **Predicted House Price: ${prediction:,.2f}**")
    
    if prediction > 200000:
        st.info("💰 This is a high-value property")
    elif prediction > 100000:
        st.info("🏡 This is a moderately-priced property")
    else:
        st.info("💡 This is an affordable property")