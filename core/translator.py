from predictor import GesturePredictor
from processor import MediaPipeProcessor
from text_processor import TextProcessor

class SignLanguageTranslator:
    def __init__(self, model_path: str, actions, mediapipe_confidence: float = 0.75, language = "es"):
        """The generic class for a sing language translator

        Args:
            model_path (str): Where the keras model is.
            actions (ndarray): A numpy array containing the model's known words.
            mediapipe_confidence (float, optional): The confidence of the mediapipe landmark detection. Defaults to 0.75.
            language (str, optional): The language's code that will be checked to correct
            the text. Defaults to "es".
        """
        
        self.processor = MediaPipeProcessor(mediapipe_confidence)
        self.predictor = GesturePredictor(model_path, actions)
        self.text_processor = TextProcessor(language)
        
    def process_frame(self, frame):
        # Orchestrates the pipeline
        keypoints = self.processor.extract_keypoints(frame)
        prediction = self.predictor.predict(keypoints)
        return prediction