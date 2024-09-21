import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load the model when the module is imported
model_filename = 'gradient_boosting_regressor_model.pkl'

def load_model():
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
gbr = load_model()

def predict(input_data):
    """
    Predict the target using the trained model.
    
    Parameters:
    input_data (list): List containing the input features, should be a 2D list like [[f1, f2, f3, f4, f5, f6, f7]]
    
    Returns:
    list: Predicted values
    """
    return gbr.predict(input_data).tolist()
