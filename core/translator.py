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
        

    def translate_video(self, video_input: str | int = 0, in_real_time: bool = True):
        """Translates the given video from CSL to spanish (currently).

        Args:
            video_input (str | int, optional): The path of the video file to be translated or 0 for the webcam. Defaults to 0.
            in_real_time (bool, optional): Whether if the translation should be displayed as it's translated or after the video is processed entirely.
            Note that this variable will be True if the video_input is the webcam.
        """
        # set the parameter to true if the webcam is being translated
        in_real_time = True if video_input == 0 else in_real_time