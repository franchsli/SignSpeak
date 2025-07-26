from numpy import ndarray
from keras.models import load_model, Model
class SignPredictor:
    def __init__(self):
        """Sign prediction class.
        """
        self.loaded_models = {}
        self.loaded_models_signs = {}
    
    def load_model(self, signs: ndarray, model_name: str, model_path: str = "models/model.keras"):
        """Loads the model in the given path in the class' memory for later use.

        Args:
            signs (ndarray): An array containing the models known signs (could be words or letters).
            model_name (str): The name that will be given to the model in memory.
            model_path (str, optional): Where the model is. Defaults to "models/model.keras".
        """
        self.loaded_models[model_name] = load_model(model_path)
        self.loaded_models_signs[model_name] = signs
    
    def get_model(self, model_name: str) -> tuple[Model, str]:
        """Returns the model stored in the class with the given name
        and its signs if found.

        Args:
            model_name (str): How the model was called when loaded.

        Raises:
            ValueError: Raised if no model with the given name is found.

        Returns:
            tuple[Model, str]: The found Model and its signs.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_models_signs[model_name]
        else:
            raise ValueError(f"Model '{model_name}' isn't loaded. Load it first and try again.")




