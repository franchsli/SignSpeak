from numpy import ndarray
from keras.models import load_model, Model
class GesturePredictor:
    def __init__(self, actions: ndarray, model_path: str = "models/model.keras"):
        """Class to predict gestures

        Args:
            actions (ndarray): A numpy array containing the model's known
            words.
            model_path (str, optional): Where the keras model is.
        """

        self.model: Model = load_model(model_path)
        self.actions = actions
    
    """
    def predict(self, keypoints):
        # Convert keypoints list to a numpy array
        keypoints = np.array(keypoints)
        # Make a prediction on the keypoints using the loaded model
        return self.model.predict(keypoints[np.newaxis, :, :])
    """
