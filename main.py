import google.generativeai as genai
from config import SETTINGS


genai.configure(api_key=SETTINGS['API_KEY'])

for model in genai.list_models():
    print(model.name)