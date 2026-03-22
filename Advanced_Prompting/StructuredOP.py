"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                  STRUCTURED GEMINI API CHATBOX PROJECT                         ║
║                     Organized Format: Explanation + Code                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    This project demonstrates building a fully-functional chatbox using Google's
    Gemini API with clean, structured code that's easy to understand and extend.

STRUCTURED FORMAT:
    1. Section Header: Clear topic
    2. Explanation: "WHY THIS SECTION?"
    3. Python Code: Implementation
    4. User Example: Real input/output
    5. Key Takeaways: Learning points

═══════════════════════════════════════════════════════════════════════════════════
"""

import os
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import google.generativeai as genai
from dotenv import load_dotenv


# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 1: DATA STRUCTURES                                                     ║
# ╚════════════════════════════════════════════════════════════════════════════════╝

"""
EXPLANATION:
    ─────────────────────────────────────────────────────────────────────────────
    WHY THIS SECTION?
        • Organize data in structured, reusable formats
        • Make code more maintainable and type-safe
        • Enable better IDE autocomplete and error checking
        • Clear separation between data and behavior
    
    WHAT WE'RE DOING:
        1. Define a Message class to store chat messages
        2. Define statuses for conversation states
        3. Create configuration data structure
    ─────────────────────────────────────────────────────────────────────────────
"""


class MessageRole(Enum):
    """Enumeration for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    STRUCTURE: Message Data Class
    
    PURPOSE: Store a single chat message with metadata
    
    FIELDS:
        role       → Who sent the message (user/assistant)
        content    → The actual message text
        timestamp  → When message was created
        tokens     → Number of tokens (optional)
    """
    role: MessageRole
    content: str
    timestamp: str
    tokens: int = 0
    
    def __str__(self) -> str:
        """Display message in readable format."""
        emoji = "👤" if self.role == MessageRole.USER else "🤖"
        return f"{emoji} [{self.timestamp}] {self.role.value.upper()}: {self.content[:60]}..."
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "tokens": self.tokens
        }


@dataclass
class ChatConfig:
    """
    STRUCTURE: Configuration Data Class
    
    PURPOSE: Store chatbox configuration
    
    FIELDS:
        model_name    → Which Gemini model to use
        temperature   → Creativity level (0.0-1.0)
        max_history   → Number of messages to keep
    """
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_history: int = 50
    system_prompt: str = "You are a helpful AI assistant."
    
    def __repr__(self) -> str:
        return f"ChatConfig(model={self.model_name}, temp={self.temperature})"


# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 2: UTILITY FUNCTIONS                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════════╝

"""
EXPLANATION:
    ─────────────────────────────────────────────────────────────────────────────
    WHY THIS SECTION?
        • Reusable functions for common tasks
        • Easier to test and maintain
        • Keep main class clean and focused
        • Single responsibility principle
    
    WHAT WE'RE DOING:
        1. Timestamp utility
        2. Input validation
        3. Error formatting
    ─────────────────────────────────────────────────────────────────────────────
"""


def get_timestamp() -> str:
    """
    FUNCTION: Get Current Timestamp
    
    PURPOSE: Standardized timestamp for all messages
    
    RETURNS:
        str → Formatted time (HH:MM:SS)
    
    EXAMPLE:
        >>> get_timestamp()
        '14:35:42'
    """
    return datetime.now().strftime("%H:%M:%S")


def validate_input(text: str) -> bool:
    """
    FUNCTION: Validate User Input
    
    PURPOSE: Ensure input meets requirements
    
    VALIDATION RULES:
        • Not empty after stripping
        • Not just whitespace
        • Valid length
    
    EXAMPLE:
        >>> validate_input("   ")
        False
        >>> validate_input("Hello AI")
        True
    """
    if not text:
        return False
    if not text.strip():
        return False
    if len(text.strip()) < 1:
        return False
    return True


def load_api_key() -> Optional[str]:
    """
    FUNCTION: Load API Key from Environment
    
    PURPOSE: Securely retrieve Gemini API key
    
    PROCESS:
        1. Load .env file
        2. Get GEMINI_API_KEY variable
        3. Return key or None
    
    RETURNS:
        str or None → API key if found
    
    EXAMPLE:
        >>> api_key = load_api_key()
        >>> if api_key:
        ...     print("✅ API key loaded")
    """
    load_dotenv()
    return os.getenv("GEMINI_API_KEY")


# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 3: CORE CHATBOX CLASS                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════════╝

"""
EXPLANATION:
    ─────────────────────────────────────────────────────────────────────────────
    WHY THIS SECTION?
        • Main functionality for chatbox
        • Encapsulate API interactions
        • Manage conversation state
        • Handle user communication
    
    WHAT WE'RE DOING:
        1. Initialize Gemini API
        2. Send and receive messages
        3. Maintain conversation history
        4. Provide user interface
    ─────────────────────────────────────────────────────────────────────────────
"""


class StructuredChatbox:
    """
    CLASS: StructuredChatbox
    
    PURPOSE: Main chatbox implementation with Gemini API
    
    KEY METHODS:
        __init__()          → Initialize chatbox
        send_message()      → Send message and get response
        chat_interactive()  → Run interactive chat loop
        get_history()       → Access conversation history
        clear_history()     → Reset conversation
    """
    
    def __init__(self, config: Optional[ChatConfig] = None):
        """
        METHOD: __init__
        
        EXPLANATION:
            ─────────────────────────────────────────────────────────────────
            Initialize the chatbox with configuration
            
            PROCESS:
                1. Set configuration (use default or provided)
                2. Load API key
                3. Configure Gemini API
                4. Initialize model
                5. Start chat session
                6. Initialize history list
            
            USER EXAMPLE:
                # Create with default config
                chatbox = StructuredChatbox()
                
                # Create with custom config
                config = ChatConfig(model_name="gemini-pro", temperature=0.5)
                chatbox = StructuredChatbox(config)
            ─────────────────────────────────────────────────────────────────
        
        PARAMETERS:
            config (ChatConfig) → Optional configuration object
        
        RAISES:
            ValueError → If API key is missing
        """
        
        # Step 1: Set configuration
        self.config = config or ChatConfig()
        print(f"⚙️  Configuration: {self.config}")
        
        # Step 2: Load API key
        print("🔑 Loading API key...")
        api_key = load_api_key()
        if not api_key:
            raise ValueError(
                "❌ API Key not found!\n"
                "   Please add GEMINI_API_KEY to your .env file"
            )
        print("✅ API key loaded successfully")
        
        # Step 3: Configure Gemini
        print("⚙️  Configuring Gemini API...")
        genai.configure(api_key=api_key)
        
        # Step 4: Initialize model
        print(f"🤖 Initializing {self.config.model_name}...")
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            system_instruction=self.config.system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.temperature
            )
        )
        
        # Step 5: Start chat session
        print("💬 Starting chat session...")
        self.chat = self.model.start_chat(history=[])
        
        # Step 6: Initialize history
        self.history: List[Message] = []
        print("✅ Chatbox initialized successfully!\n")
    
    def send_message(self, user_input: str) -> Optional[str]:
        """
        METHOD: send_message
        
        EXPLANATION:
            ─────────────────────────────────────────────────────────────────
            Send a message to the AI and get a response
            
            PROCESS:
                1. Validate user input
                2. Add user message to history
                3. Send to Gemini API
                4. Receive response
                5. Add response to history
                6. Return response text
            
            USER EXAMPLE:
                chatbox = StructuredChatbox()
                
                # Send a message
                response = chatbox.send_message("What is Python?")
                
                # Output:
                # "Python is a high-level programming language..."
            ─────────────────────────────────────────────────────────────────
        
        PARAMETERS:
            user_input (str) → The user's message
        
        RETURNS:
            str or None → Response text or None if error
        """
        
        # Step 1: Validate input
        if not validate_input(user_input):
            print("⚠️  Invalid input. Please type something.")
            return None
        
        user_input = user_input.strip()
        
        # Step 2: Add user message to history
        timestamp = get_timestamp()
        user_msg = Message(
            role=MessageRole.USER,
            content=user_input,
            timestamp=timestamp
        )
        self.history.append(user_msg)
        print(f"📤 Sent: {user_input[:50]}...")
        
        # Step 3-4: Send to API and receive response
        try:
            print("⏳ Waiting for response...")
            response = self.chat.send_message(user_input)
            response_text = response.text
            
            # Step 5: Add response to history
            timestamp = get_timestamp()
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=response_text,
                timestamp=timestamp
            )
            self.history.append(assistant_msg)
            print("📥 Response received")
            
            # Step 6: Return response
            return response_text
            
        except Exception as error:
            print(f"❌ Error communicating with API: {error}")
            return None
    
    def get_history(self) -> List[Message]:
        """
        METHOD: get_history
        
        EXPLANATION:
            ─────────────────────────────────────────────────────────────────
            Retrieve conversation history
            
            USER EXAMPLE:
                chatbox = StructuredChatbox()
                chatbox.send_message("Hello")
                chatbox.send_message("How are you?")
                
                history = chatbox.get_history()
                for msg in history:
                    print(msg)
                
                # Output:
                # 👤 [14:30:45] USER: Hello
                # 🤖 [14:30:46] ASSISTANT: Hi! I'm doing well...
                # 👤 [14:31:00] USER: How are you?
                # 🤖 [14:31:01] ASSISTANT: I'm an AI...
            ─────────────────────────────────────────────────────────────────
        
        RETURNS:
            List[Message] → All messages in history
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """
        METHOD: clear_history
        
        EXPLANATION:
            ─────────────────────────────────────────────────────────────────
            Clear conversation history and start fresh
            
            USER EXAMPLE:
                chatbox = StructuredChatbox()
                chatbox.send_message("What is AI?")
                print(f"Messages: {len(chatbox.history)}")  # Output: 2
                
                chatbox.clear_history()
                print(f"Messages: {len(chatbox.history)}")  # Output: 0
            ─────────────────────────────────────────────────────────────────
        """
        self.history.clear()
        self.chat = self.model.start_chat(history=[])
        print("✅ History cleared")
    
    def chat_interactive(self) -> None:
        """
        METHOD: chat_interactive
        
        EXPLANATION:
            ─────────────────────────────────────────────────────────────────
            Start interactive chat loop with user
            
            LOOP PROCESS:
                1. Display welcome message
                2. Get user input
                3. Check for special commands
                4. Send message if valid
                5. Display response
                6. Repeat until exit
            
            SPECIAL COMMANDS:
                • 'history'  → Show conversation history
                • 'clear'    → Clear conversation
                • 'stats'    → Show statistics
                • 'help'     → Show help
                • 'exit'     → End chat
            
            USER EXAMPLE (SCREENSHOT):
                ════════════════════════════════════════════════════════════
                🚀 STRUCTURED GEMINI CHATBOX
                ════════════════════════════════════════════════════════════
                
                👤 You: What are the benefits of Python?
                
                🤖 Assistant:
                Python has many benefits:
                1. Easy to learn syntax
                2. Large standard library
                3. Great for data science
                4. Community support
                
                👤 You: history
                
                📋 HISTORY
                ─────────────────────────────────────────────────
                [1] 👤 USER: What are the benefits of Python?
                [2] 🤖 ASSISTANT: Python has many benefits...
                
                👤 You: exit
                
                👋 Goodbye!
                ════════════════════════════════════════════════════════════
            ─────────────────────────────────────────────────────────────────
        """
        
        self._print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for special commands
                if user_input.lower() in {"exit", "quit"}:
                    self._print_goodbye()
                    break
                
                if user_input.lower() == "history":
                    self._display_history()
                    continue
                
                if user_input.lower() == "clear":
                    self.clear_history()
                    continue
                
                if user_input.lower() == "stats":
                    self._display_stats()
                    continue
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                # Send message
                response = self.send_message(user_input)
                if response:
                    print(f"\n🤖 Assistant:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Chat interrupted")
                self._print_goodbye()
                break
            except Exception as error:
                print(f"❌ Error: {error}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods for interactive chat
    # ─────────────────────────────────────────────────────────────────────────
    
    def _display_history(self) -> None:
        """Display formatted conversation history."""
        if not self.history:
            print("\n📋 No messages yet")
            return
        
        print("\n" + "="*70)
        print("📋 CONVERSATION HISTORY")
        print("="*70)
        for i, msg in enumerate(self.history, 1):
            print(f"\n[{i}] {msg}")
        print("\n" + "="*70)
    
    def _display_stats(self) -> None:
        """Display conversation statistics."""
        user_msgs = sum(1 for m in self.history if m.role == MessageRole.USER)
        ai_msgs = sum(1 for m in self.history if m.role == MessageRole.ASSISTANT)
        
        print("\n" + "="*70)
        print("📊 CONVERSATION STATISTICS")
        print("="*70)
        print(f"Total messages: {len(self.history)}")
        print(f"User messages: {user_msgs}")
        print(f"AI messages: {ai_msgs}")
        print(f"Model: {self.config.model_name}")
        print("="*70)
    
    @staticmethod
    def _print_welcome() -> None:
        """Display welcome message."""
        print("\n" + "="*70)
        print("🚀 STRUCTURED GEMINI CHATBOX")
        print("="*70)
        print("\n💡 Commands: history | clear | stats | help | exit")
        print("="*70 + "\n")
    
    @staticmethod
    def _print_goodbye() -> None:
        """Display goodbye message."""
        print("\n" + "="*70)
        print("👋 Thank you for using Structured Chatbox!")
        print("="*70 + "\n")
    
    @staticmethod
    def _print_help() -> None:
        """Display help message."""
        help_text = """
╔════════════════════════════════════════════════════════════════════╗
║                        COMMAND HELP                                ║
╠════════════════════════════════════════════════════════════════════╣
║ [any text]    │ Send message to AI                                 ║
║ history       │ View conversation history                          ║
║ clear         │ Clear conversation and start fresh                 ║
║ stats         │ Show conversation statistics                       ║
║ help          │ Show this help message                             ║
║ exit / quit   │ End the chat session                               ║
╚════════════════════════════════════════════════════════════════════╝
        """
        print(help_text)


# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 4: USAGE EXAMPLES                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════════╝

"""
EXAMPLE 1: SIMPLE MESSAGE
═════════════════════════════════════════════════════════════════════════════════

CODE:
    from StructuredOP import StructuredChatbox
    
    chatbox = StructuredChatbox()
    response = chatbox.send_message("What is machine learning?")
    print(response)

INPUT:
    "What is machine learning?"

OUTPUT:
    Machine learning is a branch of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly
    programmed. It involves feeding data to algorithms that can identify
    patterns and make predictions...

═════════════════════════════════════════════════════════════════════════════════


EXAMPLE 2: INTERACTIVE CONVERSATION
═════════════════════════════════════════════════════════════════════════════════

CODE:
    from StructuredOP import StructuredChatbox
    
    chatbox = StructuredChatbox()
    chatbox.chat_interactive()

INTERACTION FLOW:

    ══════════════════════════════════════════════════════════════════════════
    🚀 STRUCTURED GEMINI CHATBOX
    ══════════════════════════════════════════════════════════════════════════
    
    💡 Commands: history | clear | stats | help | exit
    ══════════════════════════════════════════════════════════════════════════
    
    👤 You: Explain quantum computing
    📤 Sent: Explain quantum computing...
    ⏳ Waiting for response...
    📥 Response received
    
    🤖 Assistant:
    Quantum computing is a revolutionary computing paradigm that harnesses
    quantum mechanical phenomena like superposition and entanglement. Unlike
    classical bits (0 or 1), quantum bits (qubits) can exist in both states
    simultaneously, enabling exponential computational speedup for certain
    problems like cryptography, optimization, and molecular simulation.
    
    👤 You: What are its challenges?
    
    🤖 Assistant:
    Key challenges in quantum computing include:
    1. Quantum Decoherence: Qubits lose quantum properties
    2. Error Rates: Quantum operations are error-prone
    3. Temperature: Most systems require near absolute zero
    4. Limited Qubits: Current systems have ~100-1000 qubits
    5. Algorithm Development: Few practical quantum algorithms exist
    
    👤 You: history
    
    ══════════════════════════════════════════════════════════════════════════
    📋 CONVERSATION HISTORY
    ══════════════════════════════════════════════════════════════════════════
    
    [1] 👤 [14:35:42] USER: Explain quantum computing
    [2] 🤖 [14:35:44] ASSISTANT: Quantum computing is a revolutionary...
    [3] 👤 [14:36:00] USER: What are its challenges?
    [4] 🤖 [14:36:02] ASSISTANT: Key challenges in quantum computing...
    
    ══════════════════════════════════════════════════════════════════════════
    
    👤 You: stats
    
    ══════════════════════════════════════════════════════════════════════════
    📊 CONVERSATION STATISTICS
    ══════════════════════════════════════════════════════════════════════════
    Total messages: 4
    User messages: 2
    AI messages: 2
    Model: gemini-2.5-flash
    ══════════════════════════════════════════════════════════════════════════
    
    👤 You: exit
    
    ══════════════════════════════════════════════════════════════════════════
    👋 Thank you for using Structured Chatbox!
    ══════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════════════


EXAMPLE 3: CUSTOM CONFIGURATION
═════════════════════════════════════════════════════════════════════════════════

CODE:
    from StructuredOP import StructuredChatbox, ChatConfig
    
    # Create custom configuration
    custom_config = ChatConfig(
        model_name="gemini-2.5-flash",
        temperature=0.3,  # More deterministic
        max_history=100,
        system_prompt="You are an expert Python programmer."
    )
    
    # Initialize with custom config
    chatbox = StructuredChatbox(custom_config)
    response = chatbox.send_message("How to write clean Python code?")
    print(response)

OUTPUT:
    Clean Python code principles:
    1. Follow PEP 8 style guide
    2. Use meaningful variable names
    3. Write docstrings for functions
    4. Keep functions small and focused
    5. Use type hints for clarity
    6. Write unit tests
    7. Avoid deep nesting
    8. Use context managers for resources

═════════════════════════════════════════════════════════════════════════════════


EXAMPLE 4: WORKING WITH HISTORY
═════════════════════════════════════════════════════════════════════════════════

CODE:
    from StructuredOP import StructuredChatbox
    
    chatbox = StructuredChatbox()
    
    # Send multiple messages
    chatbox.send_message("What is Python?")
    chatbox.send_message("Why use Python?")
    chatbox.send_message("Python web frameworks?")
    
    # Get and process history
    history = chatbox.get_history()
    
    print("\\n📊 USER QUESTIONS:")
    for i, msg in enumerate(history, 1):
        if msg.role.value == "user":
            print(f"  {i}. {msg.content}")
    
    print("\\n📊 TOTAL CONVERSATION LENGTH:")
    print(f"  {len(history)} messages total")

OUTPUT:
    📊 USER QUESTIONS:
      1. What is Python?
      2. Why use Python?
      3. Python web frameworks?
    
    📊 TOTAL CONVERSATION LENGTH:
      6 messages total

═════════════════════════════════════════════════════════════════════════════════
"""


def main() -> None:
    """
    MAIN FUNCTION
    
    EXPLANATION:
        Entry point for the application
    
    PROCESS:
        1. Create chatbox
        2. Start interactive chat
        3. Handle errors gracefully
    
    EXAMPLE:
        If you run: python StructuredOP.py
        It will start an interactive chatbox session
    """
    
    try:
        print("\n" + "="*70)
        print("INITIALIZING STRUCTURED CHATBOX")
        print("="*70 + "\n")
        
        # Create default configuration chatbox
        chatbox = StructuredChatbox()
        
        # Start interactive chat
        chatbox.chat_interactive()
        
    except ValueError as error:
        print(f"\n❌ Configuration Error: {error}")
        print("\n📝 Setup Instructions:")
        print("   1. Get API key from: https://ai.google.dev/")
        print("   2. Create .env file in project directory")
        print("   3. Add: GEMINI_API_KEY=your_api_key_here")
        
    except Exception as error:
        print(f"\n❌ Error: {error}")


if __name__ == "__main__":
    main()
