from numpy import ndarray
from keras.models import load_model, Model
class GesturePredictor:
    def __init__(self):
        """Gesture prediction class.
        """
        self.loaded_models = {}
        self.loaded_models_actions = {}
    
    def load_model(self, actions: ndarray, model_name: str, model_path: str = "models/model.keras"):
        """Loads the model in the given path in the class' memory for later use.

        Args:
            actions (ndarray): An array containing the models known actions (could be words or letters).
            model_name (str): The name that will be given to the model in memory.
            model_path (str, optional): Where the model is. Defaults to "models/model.keras".
        """
        self.loaded_models[model_name] = load_model(model_path)
        self.loaded_models_actions[model_name] = actions
    
    def get_model(self, model_name: str) -> tuple[Model, str]:
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_models_actions[model_name]
        else:
            raise ValueError(f"Model '{model_name}' isn't loaded. Load it first and try again.")




