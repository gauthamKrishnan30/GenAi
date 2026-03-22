"""
ChainOfThought Chatbox using Gemini API
========================================
This module implements a chatbox with chain-of-thought reasoning capabilities
using Google's Gemini API. It breaks down complex problems into step-by-step
thinking before providing final answers.

Key Features:
- Chain-of-thought reasoning for complex queries
- Step-by-step problem decomposition
- Optimized token usage
- Error handling and validation
- Session management with conversation history
"""

import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str
    content: str
    timestamp: str

    def __repr__(self) -> str:
        return f"[{self.timestamp}] {self.role}: {self.content[:50]}..."


class ChainOfThoughtChatbox:
    """
    A chatbox implementation with chain-of-thought reasoning.
    
    This class provides:
    1. Structured problem decomposition
    2. Step-by-step reasoning
    3. Optimized API usage
    4. Conversation history management
    """

    # System prompt that encourages chain-of-thought reasoning
    SYSTEM_PROMPT = """You are a helpful AI assistant with strong reasoning capabilities.

When answering questions, follow this structure:
1. **Understand**: Clarify what's being asked
2. **Analyze**: Break down the problem into components
3. **Reason**: Work through the logic step-by-step
4. **Conclude**: Provide a clear final answer

Format your response as:
STEPS:
1. [First step reasoning]
2. [Second step reasoning]
... (continue as needed)

ANSWER: [Your final answer]

Be concise but thorough in your reasoning."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the chainof-thought chatbox.
        
        Args:
            model_name: The Gemini model to use (default: gemini-2.5-flash)
            
        Raises:
            ValueError: If GEMINI_API_KEY environment variable is not set
        """
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Add it to .env file or environment variables."
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.SYSTEM_PROMPT
        )
        self.chat = self.model.start_chat(history=[])
        self.conversation_history: list[ChatMessage] = []
        self.message_count = 0

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in formatted string."""
        return datetime.now().strftime("%H:%M:%S")

    def _store_message(self, role: str, content: str) -> None:
        """
        Store message in conversation history.
        
        Args:
            role: Either 'user' or 'assistant'
            content: The message content
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=self._get_timestamp()
        )
        self.conversation_history.append(message)

    def send_message(self, user_input: str) -> Optional[str]:
        """
        Send a message and get a response with chain-of-thought reasoning.
        
        Args:
            user_input: The user's question or message
            
        Returns:
            The assistant's response with reasoning steps, or None if error
        """
        user_input = user_input.strip()
        
        if not user_input:
            return None
        
        # Store user message
        self._store_message("user", user_input)
        self.message_count += 1
        
        try:
            # Send request to Gemini API with implicit chain-of-thought
            response = self.chat.send_message(user_input)
            assistant_response = response.text
            
            # Store assistant response
            self._store_message("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as error:
            error_message = f"Error: {str(error)}"
            self._store_message("assistant", error_message)
            return None

    def chat_loop(self, show_history: bool = False) -> None:
        """
        Start an interactive chat loop.
        
        Args:
            show_history: Whether to show command options
        """
        self._print_welcome()
        
        while True:
            try:
                user_input = input("\n📝 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in {"exit", "quit"}:
                    self._print_goodbye()
                    break
                
                if user_input.lower() == "history":
                    self._display_history()
                    continue
                
                if user_input.lower() == "clear":
                    self._clear_conversation()
                    continue
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                # Send message and get response
                response = self.send_message(user_input)
                
                if response:
                    print(f"\n🤖 Assistant:\n{response}")
                else:
                    print("\n❌ Failed to get response. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Chat interrupted by user.")
                self._print_goodbye()
                break
            except Exception as error:
                print(f"\n❌ Unexpected error: {error}")

    def _display_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            print("\n📋 No conversation history yet.")
            return
        
        print("\n" + "="*50)
        print("📋 CONVERSATION HISTORY")
        print("="*50)
        
        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if msg.role == "user" else "🤖"
            print(f"\n[{i}] {role_emoji} {msg.role.upper()} ({msg.timestamp})")
            print(f"    {msg.content[:100]}...")
        
        print(f"\n📊 Total messages: {len(self.conversation_history)}")
        print("="*50)

    def _clear_conversation(self) -> None:
        """Clear conversation history and start fresh."""
        self.conversation_history.clear()
        self.chat = self.model.start_chat(history=[])
        self.message_count = 0
        print("\n✅ Conversation cleared. Starting fresh!")

    def get_summary(self) -> dict:
        """
        Get a summary of the chat session.
        
        Returns:
            Dictionary with session statistics
        """
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for m in self.conversation_history if m.role == "user"),
            "assistant_messages": sum(1 for m in self.conversation_history if m.role == "assistant"),
            "model": self.model_name,
            "session_duration": "Real-time tracking"
        }

    @staticmethod
    def _print_welcome() -> None:
        """Print welcome message."""
        print("\n" + "="*60)
        print("🚀 CHAIN-OF-THOUGHT CHATBOX WITH GEMINI API")
        print("="*60)
        print("\n✨ Features:")
        print("  • Step-by-step reasoning (Chain-of-Thought)")
        print("  • Conversation history tracking")
        print("  • Error handling and validation")
        print("\n💡 Commands:")
        print("  • Type your question to get a reasoned response")
        print("  • 'history' - View conversation history")
        print("  • 'clear' - Clear current conversation")
        print("  • 'help' - Show this help message")
        print("  • 'exit' or 'quit' - End the chat")
        print("="*60 + "\n")

    @staticmethod
    def _print_goodbye() -> None:
        """Print goodbye message."""
        print("\n" + "="*60)
        print("👋 Thank you for using Chain-of-Thought Chatbox!")
        print("="*60 + "\n")

    @staticmethod
    def _print_help() -> None:
        """Print help message."""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║                    AVAILABLE COMMANDS                       ║
╠════════════════════════════════════════════════════════════╣
║ history              │ Display full conversation history    ║
║ clear                │ Clear conversation and start fresh   ║
║ help                 │ Show this help message               ║
║ exit / quit          │ End the chat session                 ║
║ [any text]           │ Send message for chain-of-thought    ║
║                      │ reasoning                            ║
╚════════════════════════════════════════════════════════════╝
        """
        print(help_text)


def main() -> None:
    """Main entry point for the chain-of-thought chatbox."""
    try:
        # Initialize the chatbox
        chatbox = ChainOfThoughtChatbox(model_name="gemini-2.5-flash")
        
        # Start interactive chat loop
        chatbox.chat_loop()
        
        # Print session summary
        summary = chatbox.get_summary()
        print("\n📊 Session Summary:")
        for key, value in summary.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
    except ValueError as error:
        print(f"❌ Configuration Error: {error}")
        print("Please add GEMINI_API_KEY to your .env file or environment variables.")
    except Exception as error:
        print(f"❌ Fatal Error: {error}")


if __name__ == "__main__":
    main()
