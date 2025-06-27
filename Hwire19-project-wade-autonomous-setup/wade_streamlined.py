#!/usr/bin/env python3
"""
WADE Streamlined Integration for OpenHands
Clean, embedded experience with no terminal dependency
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import time

class WADEStreamlined:
    """Streamlined WADE integration for OpenHands"""
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.current_session = None
    
    async def autonomous_refactor(self, user_input: str) -> str:
        """Main autonomous refactoring function"""
        
        # Parse the request
        repo_path, vision = self._parse_request(user_input)
        
        if not repo_path:
            return self._show_help()
        
        # Show initial status
        status_msg = f"""🤖 **WADE Autonomous Refactoring Started**

📁 **Repository:** `{repo_path}`
🎯 **Vision:** {vision}
⏳ **Status:** Processing...

---
"""
        
        try:
            # Import and run the refactoring system
            from wade_refactor_system import WADERefactorSystem
            
            system = WADERefactorSystem()
            
            # Set up progress tracking
            progress_messages = []
            
            def track_progress(message, step, total):
                progress_messages.append(f"[{step}/{total}] {message}")
            
            system.set_progress_callback(track_progress)
            
            # Run the refactoring
            result = system.refactor_repository(repo_path, vision)
            
            # Generate comprehensive response
            return self._format_complete_response(result, repo_path, vision, progress_messages)
            
        except Exception as e:
            return f"""❌ **Refactoring Failed**

**Error:** {str(e)}

**Suggestion:** Try with a more specific description or check that the repository path exists.

**Example:** "Take the repo /workspace/my-project and convert it to FastAPI with async endpoints"
"""
    
    def _parse_request(self, user_input: str) -> tuple[Optional[str], str]:
        """Parse user request to extract repo path and vision"""
        
        # Check for repo refactoring intent
        triggers = ['take this repo', 'take the repo', 'refactor', 'convert', 'transform']
        if not any(trigger in user_input.lower() for trigger in triggers):
            return None, ""
        
        # Extract repo path
        import re
        path_patterns = [
            r'/[a-zA-Z0-9_/.-]+',
            r'\./[a-zA-Z0-9_/.-]+',
            r'~/[a-zA-Z0-9_/.-]+'
        ]
        
        repo_path = None
        for pattern in path_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                if os.path.exists(match) and os.path.isdir(match):
                    repo_path = match
                    break
            if repo_path:
                break
        
        # Default to workspace if no path found
        if not repo_path and os.path.exists('/workspace'):
            repo_path = '/workspace'
        
        # Extract vision
        vision = user_input.lower()
        if repo_path:
            vision = vision.replace(repo_path.lower(), '')
        
        for trigger in triggers:
            if vision.startswith(trigger):
                vision = vision[len(trigger):].strip()
                break
        
        # Clean up vision
        vision = vision.strip(' and to')
        
        return repo_path, vision or "improve the codebase structure"
    
    def _format_complete_response(self, result, repo_path: str, vision: str, progress_messages: list) -> str:
        """Format a comprehensive response with embedded elements"""
        
        if result.success:
            response = f"""✅ **WADE Autonomous Refactoring Complete!**

## 📋 Summary
{result.summary}

## 🎯 What Was Accomplished
- **Repository:** `{repo_path}`
- **Vision:** {vision}
- **Files Changed:** {len(result.files_changed + result.files_created + result.files_deleted)}

## 📁 File Changes
"""
            
            if result.files_created:
                response += f"\n**✨ Created ({len(result.files_created)}):**\n"
                for file in result.files_created:
                    response += f"- `{file}`\n"
            
            if result.files_changed:
                response += f"\n**✏️ Modified ({len(result.files_changed)}):**\n"
                for file in result.files_changed:
                    response += f"- `{file}`\n"
            
            if result.files_deleted:
                response += f"\n**🗑️ Deleted ({len(result.files_deleted)}):**\n"
                for file in result.files_deleted:
                    response += f"- `{file}`\n"
            
            # Add execution results
            if result.execution_output:
                response += f"\n## 🚀 Execution Results\n"
                try:
                    exec_data = eval(result.execution_output) if isinstance(result.execution_output, str) else result.execution_output
                    if isinstance(exec_data, dict):
                        if 'status_code' in exec_data:
                            response += f"✅ **Server Status:** HTTP {exec_data['status_code']}\n"
                            if 'response' in exec_data:
                                response += f"**Response:** `{exec_data['response'][:100]}...`\n"
                        elif 'stdout' in exec_data:
                            response += f"**Output:** `{exec_data['stdout'][:100]}...`\n"
                except:
                    response += f"**Output:** `{str(result.execution_output)[:100]}...`\n"
            
            # Add next steps
            response += f"""
## 🎯 Next Steps

1. **📝 Review Changes:** Check the modified files in your repository
2. **🧪 Test Manually:** Run and test the refactored application
3. **💾 Commit:** Save your changes to version control
4. **🚀 Deploy:** Your code is ready for deployment!

## 🔧 Quick Commands

To run your refactored application:
```bash
cd {repo_path}
python app.py  # or the main entry point
```

To view in browser (if web app):
- Local: http://localhost:8000
- OpenHands: Use the ports {self._get_available_ports()}

---
**🤖 WADE has successfully transformed your repository according to your vision!**
"""
        else:
            response = f"""❌ **Refactoring Failed**

## 📋 Summary
{result.summary}

## ❌ Errors
"""
            for error in result.errors:
                response += f"- {error}\n"
            
            response += f"""
## 🔧 Troubleshooting

1. **Check Repository Path:** Ensure `{repo_path}` exists and contains code
2. **Clarify Vision:** Try a more specific description
3. **Check Dependencies:** Ensure required packages are available

## 💡 Try Again

Example: "Take the repo {repo_path} and convert it to FastAPI with async endpoints"
"""
        
        return response
    
    def _get_available_ports(self) -> str:
        """Get available OpenHands ports"""
        return "12000, 12001"
    
    def _show_help(self) -> str:
        """Show help message"""
        return """🤖 **WADE Autonomous Repository Refactoring**

I can automatically transform your entire codebase based on your vision!

## 🎯 How to Use

Simply describe what you want to achieve:

**Examples:**
- `"Take this repo and convert it to FastAPI with async endpoints"`
- `"Refactor /workspace/my-project to use microservice architecture"`
- `"Transform this Flask app to include logging and tests"`
- `"Convert this to use clean architecture principles"`

## ✨ What I Can Do

- **🔄 Framework Conversion:** Flask → FastAPI, Django → FastAPI
- **🏗️ Architecture Refactoring:** Monolith → Microservices, MVC → Clean Architecture  
- **⚡ Modernization:** Sync → Async, Add middleware, logging, tests
- **🐳 Containerization:** Add Docker, Kubernetes configs
- **🧪 Testing:** Generate comprehensive test suites

## 📁 Supported Projects

- Python web applications (Flask, FastAPI, Django)
- JavaScript/Node.js projects
- Any structured codebase with clear entry points

## 🚀 Getting Started

1. Make sure your code is in a directory (e.g., `/workspace/my-project`)
2. Tell me what you want to achieve
3. I'll analyze, refactor, test, and show you the results!

**Ready to transform your code? Just tell me what you want to build!** 🚀
"""

# Global instance
wade_streamlined = WADEStreamlined()

async def handle_streamlined_refactor(user_input: str) -> str:
    """Main handler for streamlined refactoring"""
    return await wade_streamlined.autonomous_refactor(user_input)

def main():
    """Test function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wade_streamlined.py '<refactor request>'")
        sys.exit(1)
    
    user_input = ' '.join(sys.argv[1:])
    
    async def test_run():
        result = await handle_streamlined_refactor(user_input)
        print(result)
    
    asyncio.run(test_run())

if __name__ == "__main__":
    main()