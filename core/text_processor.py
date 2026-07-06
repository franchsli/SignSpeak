from language_tool_python import LanguageTool
from language_tool_python.utils import RateLimitError

_language_tool_instances = {}
class TextProcessor:
    def __init__(self, language="es"):
        """
        Args:
            language (str, optional): The language's code that will be checked to correct
            the text. Defaults to "es".
        """
        try:
            self.language_tool = self._get_language_tool(language)
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
    
    def _get_language_tool(self, language: str) -> LanguageTool:
        if language not in _language_tool_instances:
            _language_tool_instances[language] = LanguageTool(language)
        return _language_tool_instances[language]
