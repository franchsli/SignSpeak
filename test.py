import google.generativeai as genai
from config import SETTINGS

genai.configure(api_key=SETTINGS['API_KEY'])

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("The opposite of hot is")
print(response.text)