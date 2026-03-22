"""
═══════════════════════════════════════════════════════════════════════════════
STEP-BY-STEP GEMINI API CHATBOX IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════

This module demonstrates how to build a chatbox using Google's Gemini API.

LEARNING OBJECTIVES:
1. Understand API initialization and authentication
2. Create a conversation-based chat interface
3. Handle user input and API responses
4. Implement error handling and validation
5. Manage conversation history effectively

═══════════════════════════════════════════════════════════════════════════════
"""

import os
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: SETUP AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
    Before we can use the Gemini API, we need to:
    1. Load environment variables (API key)
    2. Configure the genai library
    3. Choose the appropriate model
    4. Create a chat session

WHY THIS MATTERS:
    - API authentication ensures secure communication
    - Model selection affects response quality and cost
    - Chat sessions maintain conversation context
"""


@dataclass
class Message:
    """
    STEP 1A: Define a data structure for messages
    
    WHY: Organizing data in a structured way makes it easier to:
    - Store conversation history
    - Display messages consistently
    - Add metadata (timestamps, roles)
    
    EXAMPLE:
        Message(role="user", content="Hello", timestamp="14:30:45")
        Message(role="assistant", content="Hi there!", timestamp="14:30:46")
    """
    role: str           # Either "user" or "assistant"
    content: str        # The actual message text
    timestamp: str      # When the message was sent


class StepByStepChatbox:
    """
    STEP 1B: Main Chatbox Class
    
    This class handles all interactions with the Gemini API.
    
    CLASS STRUCTURE:
    1. __init__()          - Initialize and configure
    2. _setup_api()        - Configure API authentication
    3. send_message()      - Send message and get response
    4. chat_interactive()  - Run interactive chat loop
    5. display_history()   - Show conversation history
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        STEP 2: INITIALIZATION
        ═════════════════════════════════════════════════════════════════
        
        PROCESS:
        1. Load .env file with secrets
        2. Get API key from environment
        3. Configure genai with API key
        4. Initialize model with selected model name
        5. Start a new chat session
        
        EXAMPLE FLOW:
            Input:  __init__("gemini-2.5-flash")
            Step 1: Load .env file (if exists)
            Step 2: Get GEMINI_API_KEY from environment
            Step 3: Configure genai library
            Step 4: Create model instance
            Step 5: Start chat session
            Output: Initialized chatbox ready for use
        
        PARAMETERS:
            model_name (str): The Gemini model to use
                             Default: "gemini-2.5-flash" (fast and efficient)
        
        RAISES:
            ValueError: If GEMINI_API_KEY is not found
        """
        
        # STEP 2A: Load environment variables
        print("📦 STEP 1: Loading environment variables...")
        load_dotenv()
        
        # STEP 2B: Retrieve API key
        print("🔑 STEP 2: Retrieving API key...")
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY not found!\n"
                "   Solution: Create a .env file with: GEMINI_API_KEY=your_key_here"
            )
        
        print("✅ API key found!")
        
        # STEP 2C: Configure the API
        print("⚙️  STEP 3: Configuring Gemini API...")
        genai.configure(api_key=self.api_key)
        
        # STEP 2D: Initialize the model
        print(f"🤖 STEP 4: Initializing {model_name} model...")
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction="You are a helpful AI assistant. Provide clear, concise, and accurate responses."
        )
        
        # STEP 2E: Start a chat session
        print("💬 STEP 5: Starting chat session...")
        self.chat = self.model.start_chat(history=[])
        
        # Initialize conversation history
        self.conversation_history: List[Message] = []
        self.model_name = model_name
        
        print("✅ Chatbox initialized successfully!\n")

    @staticmethod
    def _get_timestamp() -> str:
        """
        STEP 3A: Utility function for timestamps
        
        WHY: Track when each message was sent for better organization
        
        EXAMPLE:
            Input:  _get_timestamp()
            Output: "14:30:45"
        """
        return datetime.now().strftime("%H:%M:%S")

    def send_message(self, user_input: str) -> Optional[str]:
        """
        STEP 3: SENDING AND RECEIVING MESSAGES
        ═════════════════════════════════════════════════════════════════
        
        PROCESS:
        1. Validate user input (not empty)
        2. Store user message in history
        3. Send message to Gemini API
        4. Receive response from API
        5. Store assistant response in history
        6. Return response to user
        
        EXAMPLE FLOW:
            Input:  "What is Python?"
            Step 1: Validate input ✓
            Step 2: Store in history
            Step 3: Send to API
            Step 4: API processes request
            Step 5: Receive "Python is a programming language..."
            Step 6: Store in history
            Output: "Python is a programming language..."
        
        PARAMETERS:
            user_input (str): The message to send
        
        RETURNS:
            str: The assistant's response, or None if error
        
        EXAMPLE:
            >>> chatbox.send_message("Hello!")
            "Hello! How can I assist you today?"
        """
        
        # STEP 3A: Clean and validate input
        user_input = user_input.strip()
        
        if not user_input:
            print("⚠️  Empty message. Please type something.")
            return None
        
        print(f"📤 Sending: {user_input}")
        
        # STEP 3B: Store user message in history
        timestamp = self._get_timestamp()
        self.conversation_history.append(
            Message(role="user", content=user_input, timestamp=timestamp)
        )
        
        try:
            # STEP 3C: Send message to Gemini API
            print("⏳ Waiting for response from Gemini API...")
            response = self.chat.send_message(user_input)
            
            # STEP 3D: Extract response text
            assistant_response = response.text
            
            # STEP 3E: Store assistant response in history
            timestamp = self._get_timestamp()
            self.conversation_history.append(
                Message(role="assistant", content=assistant_response, timestamp=timestamp)
            )
            
            print(f"📥 Received response")
            
            return assistant_response
            
        except Exception as error:
            print(f"❌ Error: {error}")
            return None

    def chat_interactive(self) -> None:
        """
        STEP 4: INTERACTIVE CHAT LOOP
        ═════════════════════════════════════════════════════════════════
        
        PROCESS:
        1. Display welcome message
        2. Enter infinite loop
        3. Get user input
        4. Check for special commands (history, clear, exit)
        5. Send message if not a command
        6. Display response
        7. Repeat until user exits
        
        EXAMPLE INTERACTION:
            Welcome! Type 'exit' to quit...
            
            You: What is AI?
            Bot: AI (Artificial Intelligence) is...
            
            You: history
            [Shows conversation]
            
            You: exit
            Goodbye!
        """
        
        self._print_welcome()
        
        while True:
            try:
                # STEP 4A: Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # STEP 4B: Handle special commands
                if user_input.lower() in {"exit", "quit"}:
                    self._print_goodbye()
                    break
                
                if user_input.lower() == "history":
                    self.display_history()
                    continue
                
                if user_input.lower() == "clear":
                    self._clear_conversation()
                    continue
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                # STEP 4C: Send message
                response = self.send_message(user_input)
                
                # STEP 4D: Display response
                if response:
                    print(f"\n🤖 Assistant:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Chat interrupted.")
                self._print_goodbye()
                break
            except Exception as error:
                print(f"❌ Error: {error}")

    def display_history(self) -> None:
        """
        STEP 5: DISPLAY CONVERSATION HISTORY
        ═════════════════════════════════════════════════════════════════
        
        WHY: Users need to see what they've discussed
        
        EXAMPLE OUTPUT:
            ═══════════════ CONVERSATION HISTORY ═══════════════
            [1] 14:30:45 You: What is Python?
            [2] 14:30:46 Assistant: Python is a programming language...
            [3] 14:31:00 You: Why use Python?
            [4] 14:31:02 Assistant: Python is popular because...
            ═══════════════════════════════════════════════════════
        """
        
        if not self.conversation_history:
            print("\n📋 No messages yet. Start a conversation!")
            return
        
        print("\n" + "="*70)
        print("📋 CONVERSATION HISTORY")
        print("="*70)
        
        for i, msg in enumerate(self.conversation_history, 1):
            emoji = "👤" if msg.role == "user" else "🤖"
            role_display = msg.role.upper()
            truncated = msg.content[:75].replace("\n", " ")
            
            print(f"\n[{i}] {msg.timestamp} {emoji} {role_display}")
            print(f"    {truncated}{'...' if len(msg.content) > 75 else ''}")
        
        print("\n" + "="*70 + "\n")

    def _clear_conversation(self) -> None:
        """
        STEP 6: RESET CONVERSATION
        ═════════════════════════════════════════════════════════════════
        
        WHY: Users may want to start fresh
        
        PROCESS:
        1. Clear conversation history list
        2. Create new chat session
        3. Display confirmation
        """
        
        self.conversation_history.clear()
        self.chat = self.model.start_chat(history=[])
        print("\n✅ Conversation cleared! Starting fresh.\n")

    @staticmethod
    def _print_welcome() -> None:
        """Display welcome message with instructions."""
        print("\n" + "="*70)
        print("🚀 GEMINI API CHATBOX - STEP-BY-STEP")
        print("="*70)
        print("\n📝 How to interact:")
        print("  1. Type your question or message")
        print("  2. Press Enter to send")
        print("  3. Receive response from Gemini AI")
        print("\n💡 Special Commands:")
        print("  • 'history'  - View conversation history")
        print("  • 'clear'    - Clear conversation")
        print("  • 'help'     - Show help")
        print("  • 'exit'     - End chat")
        print("="*70 + "\n")

    @staticmethod
    def _print_goodbye() -> None:
        """Display goodbye message."""
        print("\n" + "="*70)
        print("👋 Thank you for using the Gemini Chatbox!")
        print("="*70 + "\n")

    @staticmethod
    def _print_help() -> None:
        """Display detailed help."""
        help_text = """
╔════════════════════════════════════════════════════════════════════╗
║                        COMMAND HELP                                ║
╠════════════════════════════════════════════════════════════════════╣
║ [any text]    │ Send message to AI                                 ║
║ history       │ View all messages in this conversation             ║
║ clear         │ Reset conversation and start fresh                 ║
║ help          │ Show this help message                             ║
║ exit / quit   │ End the chat session                               ║
╚════════════════════════════════════════════════════════════════════╝
        """
        print(help_text)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAMPLE USAGE AND EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
"""
EXAMPLE 1: BASIC USAGE
═════════════════════════════════════════════════════════════════════

CODE:
    chatbox = StepByStepChatbox()
    response = chatbox.send_message("What is machine learning?")
    print(response)

INPUT:  "What is machine learning?"

OUTPUT: "Machine learning is a subset of artificial intelligence that 
         enables computers to learn from data without being explicitly 
         programmed. It involves training algorithms on datasets to 
         recognize patterns and make predictions..."

═════════════════════════════════════════════════════════════════════


EXAMPLE 2: INTERACTIVE CONVERSATION
═════════════════════════════════════════════════════════════════════

CODE:
    chatbox = StepByStepChatbox()
    chatbox.chat_interactive()

INTERACTION:
    ======================================================================
    🚀 GEMINI API CHATBOX - STEP-BY-STEP
    ======================================================================
    
    📝 How to interact:
      1. Type your question or message
      2. Press Enter to send
      3. Receive response from Gemini AI
    
    💡 Special Commands:
      • 'history'  - View conversation history
      • 'clear'    - Clear conversation
      • 'help'     - Show help
      • 'exit'     - End chat
    ======================================================================
    
    👤 You: Explain Python for beginners
    
    ⏳ Waiting for response from Gemini API...
    
    🤖 Assistant:
    Python is a beginner-friendly programming language known for its 
    simple, readable syntax. Here's why beginners love it:
    
    1. Easy to learn: Python reads almost like English
    2. Versatile: Used in web, data science, AI, automation
    3. Large community: Lots of tutorials and resources
    4. Libraries: Pre-built code for common tasks
    
    Example: print("Hello, World!")
    
    👤 You: What can I build with Python?
    
    🤖 Assistant:
    With Python, you can build:
    
    1. Web applications (Django, Flask)
    2. Data analysis tools
    3. Machine learning models
    4. Desktop applications
    5. Automation scripts
    6. Games
    7. AI chatbots
    
    👤 You: exit
    
    ======================================================================
    👋 Thank you for using the Gemini Chatbox!
    ======================================================================

═════════════════════════════════════════════════════════════════════


EXAMPLE 3: PROGRAMMATIC ACCESS WITH HISTORY
═════════════════════════════════════════════════════════════════════

CODE:
    chatbox = StepByStepChatbox()
    
    # Send multiple messages
    chatbox.send_message("What is Python?")
    chatbox.send_message("What's the best use case?")
    chatbox.send_message("How to learn Python?")
    
    # View conversation history
    chatbox.display_history()

OUTPUT:
    ======================================================================
    📋 CONVERSATION HISTORY
    ======================================================================
    
    [1] 10:15:30 👤 USER
        What is Python?
    
    [2] 10:15:32 🤖 ASSISTANT
        Python is a high-level programming language...
    
    [3] 10:15:45 👤 USER
        What's the best use case?
    
    [4] 10:15:47 🤖 ASSISTANT
        Python is best for data science and web development...
    
    [5] 10:16:00 👤 USER
        How to learn Python?
    
    [6] 10:16:02 🤖 ASSISTANT
        Here are the best ways to learn Python:
        1. Take online courses
        2. Practice with coding challenges
        3. Build projects...
    
    ======================================================================

═════════════════════════════════════════════════════════════════════
"""


def main() -> None:
    """
    MAIN FUNCTION - ENTRY POINT
    ═════════════════════════════════════════════════════════════════
    
    PROCESS:
    1. Try to initialize the chatbox
    2. Start interactive chat
    3. Handle errors gracefully
    
    ERROR HANDLING:
    - ValueError: Missing API key
    - Exception: Any other unexpected errors
    """
    
    try:
        print("\n" + "="*70)
        print("INITIALIZING GEMINI API CHATBOX")
        print("="*70)
        
        # Initialize chatbox
        chatbox = StepByStepChatbox(model_name="gemini-2.5-flash")
        
        # Start interactive chat
        chatbox.chat_interactive()
        
    except ValueError as error:
        print(f"\n❌ Configuration Error: {error}")
        print("\n📝 Setup Instructions:")
        print("   1. Get your Gemini API key from: https://ai.google.dev")
        print("   2. Create a .env file in the same directory")
        print("   3. Add: GEMINI_API_KEY=your_api_key_here")
        
    except Exception as error:
        print(f"\n❌ Unexpected Error: {error}")


if __name__ == "__main__":
    main()
