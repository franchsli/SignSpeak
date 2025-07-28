from language_tool_python import LanguageToolPublicAPI
from language_tool_python.utils import RateLimitError


class TextProcessor:
    def __init__(self, language="es"):
        """
        Args:
            language (str, optional): The language's code that will be checked to correct
            the text. Defaults to "es".
        """
        try:
            self.language_tool = LanguageToolPublicAPI(language)
        except RateLimitError:
            self.language_tool = None
            print(
                "You have exceeded the rate limit for the free LanguageTool API. Please try again later. The class must be re initialized later"
            )

    def correct_sentence(self, sentence: str):
        try:
            if self.language_tool:
                corrected_text = self.language_tool.correct(sentence)
                return corrected_text
            else:
                print(
                    "You have exceeded the rate limit for the free LanguageTool API. Please try again later."
                )
        except RateLimitError:
            print(
                "You have exceeded the rate limit for the free LanguageTool API. Please try again later."
            )
