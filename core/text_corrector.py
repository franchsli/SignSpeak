from language_tool_python import LanguageTool
from language_tool_python.utils import LanguageToolError

_language_tool_instances = {}


class TextCorrector:
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

    def correct_text(self, sentence: str) -> str:
        try:
            if self.language_tool:
                corrected_text = self.language_tool.correct(sentence)
                return corrected_text
            else:
                print(
                    "LanguageTool failed to initialize previously so no correction is applied."
                )
                return sentence
        except LanguageToolError as e:
            print(f"LanguageTool error:\n {e}")
            return sentence
    
    def close_language_tool(self):
        """Closes the language tool server used in the processor.
        """
        if self.language_tool:
            language = self.language_tool.language.tag
            _language_tool_instances.pop(language)
            self.language_tool.close()
            self.language_tool = None

    def _get_language_tool(self, language: str) -> LanguageTool:
        if language not in _language_tool_instances:
            _language_tool_instances[language] = LanguageTool(
                language, language_tool_download_version="6.6"
            )
        return _language_tool_instances[language]
