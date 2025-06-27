#!/usr/bin/env python3
"""
WADE OpenHands Integration
Main entry point for autonomous repository refactoring through OpenHands
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '/workspace')

from openhands_refactor_agent import handle_refactor_request, refactor_agent

class WADEOpenHandsInterface:
    """Interface between WADE and OpenHands"""
    
    def __init__(self):
        self.agent = refactor_agent
        self.session_log = []
    
    def log_interaction(self, user_input: str, response: str):
        """Log user interactions"""
        self.session_log.append({
            'timestamp': self._get_timestamp(),
            'user_input': user_input,
            'response': response[:500] + "..." if len(response) > 500 else response
        })
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and return response"""
        try:
            # Check if this is a repo refactoring request
            if self.agent.detect_repo_refactor_intent(user_input):
                print(f"🤖 WADE detected repo refactoring request: {user_input}")
                response = await handle_refactor_request(user_input)
                self.log_interaction(user_input, response)
                return response
            else:
                # Not a refactoring request
                return self._get_help_message()
                
        except Exception as e:
            error_response = f"❌ Error processing request: {str(e)}"
            self.log_interaction(user_input, error_response)
            return error_response
    
    def _get_help_message(self) -> str:
        """Get help message for users"""
        return """🤖 WADE Autonomous Repo Refactor Agent

I can help you automatically refactor repositories based on your vision!

**How to use me:**
Just describe what you want to do with your repository:

📝 **Examples:**
• "Take this repo and convert it to FastAPI with async endpoints"
• "Refactor this Flask app to use microservice architecture"
• "Transform this monolith into a modular plugin system"
• "Modify this repo to follow clean architecture principles"
• "Take the repo /path/to/project and add logging middleware"

**What I can do:**
✅ Analyze your existing codebase
✅ Understand your architectural vision
✅ Make comprehensive code changes
✅ Add tests and documentation
✅ Run tests and validate functionality
✅ Execute the refactored code
✅ Show you the results

**Supported transformations:**
• Flask → FastAPI conversion
• Monolith → Microservices
• Add logging, middleware, tests
• Restructure for clean architecture
• Add Docker configuration
• And much more!

Just tell me what you want to achieve! 🚀"""

    def get_session_summary(self) -> str:
        """Get summary of current session"""
        if not self.session_log:
            return "No interactions in this session yet."
        
        summary = f"Session Summary ({len(self.session_log)} interactions):\n"
        for i, interaction in enumerate(self.session_log, 1):
            summary += f"{i}. {interaction['timestamp']}: {interaction['user_input'][:100]}...\n"
        
        return summary

# Global interface instance
wade_interface = WADEOpenHandsInterface()

async def main_handler(user_input: str) -> str:
    """Main handler function for OpenHands integration"""
    return await wade_interface.process_user_input(user_input)

def demo_refactor():
    """Demo function to test the refactoring system"""
    async def run_demo():
        print("🚀 WADE Demo: Autonomous Repository Refactoring")
        print("=" * 60)
        
        # Demo request
        demo_request = "Take the repo /workspace/demo_repo and convert it to FastAPI with async endpoints and add logging middleware"
        
        print(f"📝 Demo Request: {demo_request}")
        print("-" * 60)
        
        result = await main_handler(demo_request)
        print(result)
        
        print("\n" + "=" * 60)
        print("🎯 Demo completed! Check /workspace/demo_repo for changes.")
    
    asyncio.run(run_demo())

def interactive_mode():
    """Interactive mode for testing"""
    async def run_interactive():
        print("🤖 WADE Interactive Mode")
        print("Type 'help' for usage instructions, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print(wade_interface._get_help_message())
                    continue
                
                if user_input.lower() == 'summary':
                    print(wade_interface.get_session_summary())
                    continue
                
                if not user_input:
                    continue
                
                print("🤖 WADE: Processing your request...")
                response = await main_handler(user_input)
                print(f"🤖 WADE: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    asyncio.run(run_interactive())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_refactor()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        else:
            # Process single command
            user_input = " ".join(sys.argv[1:])
            result = asyncio.run(main_handler(user_input))
            print(result)
    else:
        print("WADE OpenHands Integration")
        print("Usage:")
        print("  python wade_openhands_integration.py demo")
        print("  python wade_openhands_integration.py interactive")
        print("  python wade_openhands_integration.py 'your refactor request'")