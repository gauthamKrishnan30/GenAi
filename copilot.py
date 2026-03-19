"""Copilot chatbot built on Gemini (Google Generative AI API).

This module provides a minimal conversational interface that can be run
as a standalone script or imported as a class.

Requirements:
  - google-generativeai (or google.genai if upgraded)
  - python-dotenv (for loading GEMINI_API_KEY from a .env file)

Usage:
  python copilot.py

Ensure a GEMINI_API_KEY is set in your environment or in a .env file.
"""

import os
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SYSTEM_PROMPT = (
    "You are Copilot, a helpful assistant that answers user questions clearly "
    "and concisely."
)


class GeminiCopilot:
    """A small wrapper around Gemini chat.

    This class maintains a local history and can be used for interactive or
    programmatic chat flows.
    """

    def __init__(
        self,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        # Prefer a .env located in the same directory as this module (useful when
        # running from a different working directory).
        dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(dotenv_path=dotenv_path)

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} was not found. Add it to your environment or a .env file."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.system_prompt = system_prompt
        self._reset_chat()

    def _reset_chat(self) -> None:
        """Reset the chat state (clear history and start a new session)."""
        # Gemini expects history items to be in the `Content` format.
        # See: google.generativeai.types.ContentDict
        # Gemini expects roles to be either "user" or "model".
        # We use a user message for the initial instruction prompt.
        self.history: List[dict] = [
            {"role": "user", "parts": [{"text": self.system_prompt}]}
        ]
        self.chat = self.model.start_chat(history=self.history)

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Reset the conversation and optionally replace the system prompt."""
        if system_prompt:
            self.system_prompt = system_prompt
        self._reset_chat()

    def send(self, user_message: str) -> str:
        """Send a user message to Gemini and return the assistant response."""
        if not user_message:
            return ""

        response = self.chat.send_message(user_message)

        # Keep a simple history in case the caller wants to access it.
        self.history.append(
            {"role": "user", "parts": [{"text": user_message}]}
        )
        assistant_text = getattr(response, "text", str(response))
        self.history.append(
            {"role": "model", "parts": [{"text": assistant_text}]}
        )

        return assistant_text


def main() -> None:
    bot = GeminiCopilot()

    print("Copilot started. Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Copilot: Goodbye!")
            break

        try:
            answer = bot.send(user_input)
            print(f"Copilot: {answer}\n")
        except Exception as error:
            print(f"Copilot: Request failed: {error}\n")


if __name__ == "__main__":
    main()
