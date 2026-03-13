import google.generativeai as genai
from dotenv import load_dotenv
import os
from UserInput import userprompt

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model=genai.GenerativeModel("gemini-2.5-flash")

chat=model.start_chat(history=[])

print("\n****** CHATBOT HAS STARTED  ******\n")

while True:
    user_input=input("Bot: Ask questions....!:")
    if user_input.lower()=="exit":
        break
    prompt = userprompt(user_input)
    response=chat.send_message(prompt)

    print("Bot: ",response.text)