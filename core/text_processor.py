from language_tool_python import LanguageTool
from language_tool_python.utils import LanguageToolError

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
        except LanguageToolError as e:
            self.language_tool = None
            print(f"LanguageTool failed to initialize:\n {e}")

    def correct_sentence(self, sentence: str):
        try:
            if self.language_tool:
                corrected_text = self.language_tool.correct(sentence)
                return corrected_text
            else:
                print(
                    "LanguageTool failed to initialize previously so no correction is applied."
                )
        except LanguageToolError as e:
            print(f"LanguageTool failed to initialize:\n {e}")

    def _get_language_tool(self, language: str) -> LanguageTool:
        if language not in _language_tool_instances:
            _language_tool_instances[language] = LanguageTool(
                language, language_tool_download_version="6.6"
            )
        return _language_tool_instances[language]
