"""
Role-Based Prompting Chatbox using Gemini API
==============================================
This module implements a sophisticated chatbox that uses role-based prompting
to allow the Gemini AI to adopt different personas and expertise domains.

Key Features:
- Multiple AI roles (Expert, Tutor, Developer, etc.)
- Dynamic role switching during conversation
- Optimized prompt engineering
- Role-specific behaviors and constraints
- Session management with role tracking
"""

import os
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv


class AIRole(Enum):
    """Available AI roles and personas."""
    EXPERT = "expert"
    TUTOR = "tutor"
    DEVELOPER = "developer"
    ASSISTANT = "assistant"
    ANALYST = "analyst"
    CREATIVE = "creative"


@dataclass
class RolePrompt:
    """Defines a role's system prompt and characteristics."""
    role_name: str
    description: str
    system_prompt: str
    temperature: float = 0.7
    emoji: str = "🤖"

    def __str__(self) -> str:
        return f"{self.emoji} {self.role_name}: {self.description}"


@dataclass
class ChatMessage:
    """Represents a single chat message with metadata."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    ai_role: str = "assistant"
    
    def __repr__(self) -> str:
        truncated = self.content[:80].replace("\n", " ")
        return f"[{self.timestamp}] {self.role} ({self.ai_role}): {truncated}..."


class RoleManager:
    """Manages available AI roles and their configurations."""
    
    ROLES: Dict[str, RolePrompt] = {
        AIRole.EXPERT.value: RolePrompt(
            role_name="Expert Consultant",
            description="Provides authoritative, in-depth analysis with citations",
            system_prompt="""You are an Expert Consultant with deep domain knowledge.
Your responsibilities:
- Provide authoritative and well-researched answers
- Support claims with evidence and references
- Acknowledge limitations and uncertainties
- Offer multiple perspectives when relevant
- Use technical terminology appropriately

Format: Begin with the core answer, then provide supporting details and sources.""",
            emoji="🧠"
        ),
        AIRole.TUTOR.value: RolePrompt(
            role_name="Educational Tutor",
            description="Teaches concepts through guided learning",
            system_prompt="""You are an Educational Tutor dedicated to helping learners.
Your responsibilities:
- Assess the learner's current understanding first
- Break down complex topics into digestible parts
- Use analogies and real-world examples
- Ask clarifying questions
- Provide practice questions and feedback
- Encourage critical thinking

Format: Use Socratic method. Engage, explain, and encourage.""",
            emoji="👨‍🏫"
        ),
        AIRole.DEVELOPER.value: RolePrompt(
            role_name="Code Expert",
            description="Specializes in programming and software development",
            system_prompt="""You are a Senior Code Expert and Software Developer.
Your responsibilities:
- Provide clean, optimized, production-ready code
- Explain code decisions and improvements
- Follow best practices and design patterns
- Consider performance, security, and maintainability
- Suggest alternative approaches
- Provide complete code examples

Format: Lead with the solution, explain, then provide code examples with comments.""",
            emoji="👨‍💻"
        ),
        AIRole.ANALYST.value: RolePrompt(
            role_name="Data Analyst",
            description="Analyzes data and provides data-driven insights",
            system_prompt="""You are a Data Analyst specializing in insights.
Your responsibilities:
- Extract meaningful patterns from information
- Provide quantitative and qualitative analysis
- Support conclusions with data
- Identify trends and anomalies
- Offer actionable recommendations
- Visualize concepts when helpful

Format: Start with key findings, then detailed analysis.""",
            emoji="📊"
        ),
        AIRole.CREATIVE.value: RolePrompt(
            role_name="Creative Muse",
            description="Generates creative and innovative ideas",
            system_prompt="""You are a Creative Muse inspiring innovation.
Your responsibilities:
- Generate original and imaginative ideas
- Think outside conventional boundaries
- Connect disparate concepts
- Encourage creative exploration
- Provide multiple creative directions
- Balance creativity with practicality

Format: Start bold, iterate, and refine.""",
            emoji="🎨"
        ),
        AIRole.ASSISTANT.value: RolePrompt(
            role_name="Helpful Assistant",
            description="Provides friendly, comprehensive assistance",
            system_prompt="""You are a Helpful Assistant here to support the user.
Your responsibilities:
- Provide clear, friendly, and practical help
- Ask clarifying questions when needed
- Adapt your communication style to the user
- Offer step-by-step guidance
- Acknowledge when something is outside your scope
- Help users feel supported and understood

Format: Be warm, clear, and actionable.""",
            emoji="😊"
        ),
    }

    @classmethod
    def get_role(cls, role: AIRole | str) -> RolePrompt:
        """Get a role configuration by AIRole enum or string."""
        if isinstance(role, AIRole):
            role_key = role.value
        else:
            role_key = role.lower()
        
        if role_key not in cls.ROLES:
            raise ValueError(f"Unknown role: {role_key}. Available: {list(cls.ROLES.keys())}")
        
        return cls.ROLES[role_key]

    @classmethod
    def list_roles(cls) -> List[str]:
        """List all available role names."""
        return list(cls.ROLES.keys())

    @classmethod
    def get_role_descriptions(cls) -> str:
        """Get formatted descriptions of all roles."""
        descriptions = ["Available AI Roles:\n" + "="*50]
        for role_key, role_prompt in cls.ROLES.items():
            descriptions.append(f"{role_prompt.emoji} {role_prompt.role_name}")
            descriptions.append(f"   {role_prompt.description}\n")
        return "\n".join(descriptions)


class RoleBasedChatbox:
    """
    Implements a chatbox with role-based prompting capabilities.
    
    Users can switch between different AI roles/personas to get
    specialized responses tailored to each role's expertise.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", default_role: str = "assistant"):
        """
        Initialize the role-based chatbox.
        
        Args:
            model_name: The Gemini model to use
            default_role: The initial role for the chatbox
            
        Raises:
            ValueError: If API key is missing or role is invalid
        """
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Add it to .env file or environment variables."
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self._initialize_model(default_role)
        
        self.conversation_history: List[ChatMessage] = []
        self.current_role = default_role
        self.role_switch_count = 0

    def _initialize_model(self, role: str) -> None:
        """Initialize the model with a specific role's system prompt."""
        role_prompt = RoleManager.get_role(role)
        self.current_role = role
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=role_prompt.system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=role_prompt.temperature
            )
        )
        self.chat = self.model.start_chat(history=[])

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def switch_role(self, new_role: str) -> str:
        """
        Switch to a different AI role.
        
        Args:
            new_role: The role to switch to
            
        Returns:
            Confirmation message
            
        Raises:
            ValueError: If role is invalid
        """
        if new_role == self.current_role:
            return f"Already using {new_role} role."
        
        try:
            role_prompt = RoleManager.get_role(new_role)
            self._initialize_model(new_role)
            self.role_switch_count += 1
            return f"✅ Switched to {role_prompt} role."
        except ValueError as e:
            return f"❌ Error: {e}"

    def send_message(self, user_input: str) -> Optional[str]:
        """
        Send a message to the current AI role and get a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            The AI's response or None if error occurs
        """
        user_input = user_input.strip()
        
        if not user_input:
            return None
        
        # Store user message
        timestamp = self._get_timestamp()
        self.conversation_history.append(
            ChatMessage(
                role="user",
                content=user_input,
                timestamp=timestamp,
                ai_role=self.current_role
            )
        )
        
        try:
            response = self.chat.send_message(user_input)
            assistant_response = response.text
            
            # Store assistant response
            self.conversation_history.append(
                ChatMessage(
                    role="assistant",
                    content=assistant_response,
                    timestamp=self._get_timestamp(),
                    ai_role=self.current_role
                )
            )
            
            return assistant_response
            
        except Exception as error:
            return None

    def chat_loop(self) -> None:
        """Start an interactive chat loop with role-based prompting."""
        self._print_welcome()
        
        while True:
            try:
                user_input = input(f"\n📝 You ({self.current_role}): ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in {"exit", "quit"}:
                    self._print_goodbye()
                    break
                
                if user_input.lower() == "roles":
                    print("\n" + RoleManager.get_role_descriptions())
                    continue
                
                if user_input.lower().startswith("role "):
                    new_role = user_input[5:].strip().lower()
                    print(self.switch_role(new_role))
                    continue
                
                if user_input.lower() == "history":
                    self._display_history()
                    continue
                
                if user_input.lower() == "clear":
                    self._clear_conversation()
                    continue
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                if user_input.lower() == "summary":
                    self._print_summary()
                    continue
                
                # Send message and get response
                response = self.send_message(user_input)
                
                if response:
                    current_role_prompt = RoleManager.get_role(self.current_role)
                    print(f"\n{current_role_prompt.emoji} Assistant:\n{response}")
                else:
                    print("\n❌ Failed to get response. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Chat interrupted by user.")
                self._print_goodbye()
                break
            except Exception as error:
                print(f"\n❌ Unexpected error: {error}")

    def _display_history(self) -> None:
        """Display conversation history with role information."""
        if not self.conversation_history:
            print("\n📋 No conversation history yet.")
            return
        
        print("\n" + "="*70)
        print("📋 CONVERSATION HISTORY")
        print("="*70)
        
        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if msg.role == "user" else "🤖"
            print(f"\n[{i}] {role_emoji} {msg.role.upper()} ({msg.ai_role}) - {msg.timestamp}")
            print(f"    {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        
        print(f"\n📊 Total messages: {len(self.conversation_history)}")
        print("="*70)

    def _clear_conversation(self) -> None:
        """Clear conversation history and reset the model."""
        self.conversation_history.clear()
        self._initialize_model(self.current_role)
        print("\n✅ Conversation cleared! Starting fresh with the same role.")

    def _print_summary(self) -> None:
        """Print conversation summary and statistics."""
        role_counts = {}
        for msg in self.conversation_history:
            role_counts[msg.ai_role] = role_counts.get(msg.ai_role, 0) + 1
        
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)
        print(f"Total messages: {len(self.conversation_history)}")
        print(f"User messages: {sum(1 for m in self.conversation_history if m.role == 'user')}")
        print(f"Assistant messages: {sum(1 for m in self.conversation_history if m.role == 'assistant')}")
        print(f"Role switches: {self.role_switch_count}")
        print(f"Current role: {self.current_role}")
        print("\nMessages by role:")
        for role, count in sorted(role_counts.items()):
            print(f"  • {role}: {count}")
        print("="*70)

    @staticmethod
    def _print_welcome() -> None:
        """Print welcome message."""
        print("\n" + "="*70)
        print("🎭 ROLE-BASED PROMPTING CHATBOX WITH GEMINI API")
        print("="*70)
        print("\n✨ Features:")
        print("  • Multiple AI roles with specialized expertise")
        print("  • Dynamic role switching during conversation")
        print("  • Conversation history with role tracking")
        print("  • Session statistics and summaries")
        print("\n💡 Quick Commands:")
        print("  • 'roles'    - View all available roles")
        print("  • 'role [name]' - Switch to a specific role")
        print("  • 'history'  - View conversation history")
        print("  • 'summary'  - Show session statistics")
        print("  • 'clear'    - Clear conversation")
        print("  • 'help'     - Show full help")
        print("  • 'exit'     - End chat")
        print("="*70 + "\n")

    @staticmethod
    def _print_goodbye() -> None:
        """Print goodbye message."""
        print("\n" + "="*70)
        print("👋 Thank you for using Role-Based Prompting Chatbox!")
        print("="*70 + "\n")

    @staticmethod
    def _print_help() -> None:
        """Print detailed help message."""
        help_text = """
╔════════════════════════════════════════════════════════════════════╗
║                    COMMAND REFERENCE                               ║
╠════════════════════════════════════════════════════════════════════╣
║ ROLE MANAGEMENT:                                                    ║
║   roles                 │ List all available AI roles               ║
║   role [name]           │ Switch to a specific role                 ║
║                         │ Example: role developer                   ║
╠════════════════════════════════════════════════════════════════════╣
║ CONVERSATION:                                                       ║
║   [any text]            │ Send message to current role              ║
║   history               │ View full conversation history            ║
║   clear                 │ Clear history and start fresh             ║
║   summary               │ Show session statistics                   ║
╠════════════════════════════════════════════════════════════════════╣
║ GENERAL:                                                            ║
║   help                  │ Show this help message                    ║
║   exit / quit           │ End the chat session                      ║
╚════════════════════════════════════════════════════════════════════╝
        """
        print(help_text)


def main() -> None:
    """Main entry point for the role-based chatbox."""
    try:
        # Initialize with default role (assistant)
        chatbox = RoleBasedChatbox(
            model_name="gemini-2.5-flash",
            default_role="assistant"
        )
        
        # Start interactive chat loop
        chatbox.chat_loop()
        
    except ValueError as error:
        print(f"❌ Configuration Error: {error}")
        print("Please add GEMINI_API_KEY to your .env file or environment variables.")
    except Exception as error:
        print(f"❌ Fatal Error: {error}")


if __name__ == "__main__":
    main()
