from predictor import GesturePredictor
from processor import MediaPipeProcessor
from text_processor import TextProcessor

class SignLanguageTranslator:
    def __init__(self, config: dict):
        
        self.processor = MediaPipeProcessor(config["confidence"])
        self.predictor = GesturePredictor(config["model_path"], config["actions"])
        self.text_processor = TextProcessor(config["language"])
        
    def process_frame(self, frame):
        # Orchestrates the pipeline
        keypoints = self.processor.extract_keypoints(frame)
        prediction = self.predictor.predict(keypoints)
        return prediction