import google.generativeai as genai
from config import SETTINGS
import time

genai.configure(api_key=SETTINGS['API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash-exp')
chat = model.start_chat(history=[])
video_file = genai.upload_file(path=SETTINGS['TEST_FILE_NAME'])

# Check whether the file is ready to be used.
while video_file.state.name == 'PROCESSING':
    print('LOADING VIDEO...', end='\n')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)
    

if video_file.state.name == 'FAILED':
  raise ValueError(video_file.state.name)

else:
    print('VIDEO LOADED')
    prompt = input('')
    response = chat.send_message([video_file, prompt],
                                  request_options={'timeout': 600})
    while len(chat.history) <= 20:
        response = chat.send_message(input(''))
        print(response.text)

genai.delete_file(video_file.name)
print('VIDEO DELETED')