from numpy import ndarray
from keras.models import load_model, Model
class GesturePredictor:
    def __init__(self):
        """Class to predict gestures.

        Args:
            actions (ndarray): A numpy array containing the model's known
            words.
            model_path (str, optional): Where the keras model is.
        """
        self.loaded_models = {}
        self.loaded_models_actions = {}
    
    def load_model(self, actions: list[str], model_name: str, model_path: str = "models/model.keras"):
        self.loaded_models[model_name] = load_model(model_path)
        self.loaded_models_actions[model_name] = actions
        

