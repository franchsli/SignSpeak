from numpy import ndarray
from keras.models import load_model, Model
class GesturePredictor:
    def __init__(self):
        """Class to predict gestures.
        """
        self.loaded_models = {}
        self.loaded_models_actions = {}
    
    def load_model(self, actions: ndarray, model_name: str, model_path: str = "models/model.keras"):
        self.loaded_models[model_name] = load_model(model_path)
        self.loaded_models_actions[model_name] = actions



