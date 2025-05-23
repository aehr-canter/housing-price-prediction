from buildapp import *

def main():

    # Page configuration
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide"
    )
    
    # Title and description
    st.title("🏠 House Price Prediction App")
    st.markdown("Enter the house features below to get an estimated price prediction.")
    
    # Load model components
    model, preprocessor = load_model_components()
    
    # Create input sections
    user_inputs = create_user_inputs()
    advanced_inputs = create_advanced_inputs()
    
    # Create input DataFrame
    input_data = create_input_dataframe(user_inputs, advanced_inputs)
    
    # Show input summary
    if st.checkbox("Show Input Summary"):
        st.subheader("Input Summary")
        # Show only user-modified features
        display_features = {**user_inputs, **advanced_inputs}
        st.json(display_features)
    
    # Prediction button and result
    if st.button("🔮 Predict House Price", type="primary"):
        with st.spinner("Making prediction..."):
            prediction = make_prediction(model, preprocessor, input_data)
            display_prediction_result(prediction)
    
    # Additional information
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This app uses a machine learning model trained on housing data to predict house prices.
        
        **Key Features:**
        - Basic house characteristics (lot area, quality, year built, etc.)
        - Advanced options for detailed customization
        - Real-time price prediction
        
        **How to use:**
        1. Adjust the house features using the input controls
        2. Optionally expand advanced options for more detailed inputs
        3. Click 'Predict House Price' to get your estimate
        
        **Note:** Predictions are estimates based on historical data and should not be used as the sole basis for real estate decisions.
        """)


if __name__ == "__main__":
    main()