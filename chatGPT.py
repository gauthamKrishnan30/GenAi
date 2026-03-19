import os

import google.generativeai as genai
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Add it to your environment or a .env file."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    chat = model.start_chat(history=[])

    print("\nGemini chatbot started.")
    print("Type 'exit' to close the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break

        try:
            response = chat.send_message(user_input)
            print(f"Bot: {response.text}\n")
        except Exception as error:
            print(f"Bot: Request failed: {error}\n")


if __name__ == "__main__":
    main()
