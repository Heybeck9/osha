#!/usr/bin/env python3
"""
OpenHands Autonomous Repo Refactor Agent
Integrates WADE refactoring system with OpenHands frontend
"""

import os
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from wade_refactor_system import WADERefactorSystem, RefactorResult

class OpenHandsRefactorAgent:
    """Agent that handles repo refactoring through OpenHands interface"""
    
    def __init__(self):
        self.wade_system = WADERefactorSystem()
        self.current_task = None
        self.progress_log = []
        
        # Set up progress callback
        self.wade_system.set_progress_callback(self._log_progress)
    
    def _log_progress(self, message: str, step: int, total_steps: int):
        """Log progress for OpenHands UI"""
        progress_entry = {
            'step': step,
            'total_steps': total_steps,
            'message': message,
            'timestamp': self._get_timestamp()
        }
        self.progress_log.append(progress_entry)
        print(f"[PROGRESS {step}/{total_steps}] {message}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def detect_repo_refactor_intent(self, user_input: str) -> bool:
        """Detect if user wants to refactor a repository"""
        triggers = [
            'take this repo',
            'take the repo',
            'refactor repo',
            'refactor this repo',
            'refactor the repo',
            'modify repo',
            'modify this repo',
            'modify the repo',
            'edit repo to',
            'edit this repo',
            'edit the repo',
            'transform repo',
            'transform this repo',
            'transform the repo',
            'align repo with',
            'repo vision',
            'change this codebase',
            'change the codebase',
            'update this project',
            'update the project',
            'convert this repo',
            'convert the repo',
            'convert this to',
            'refactor to'
        ]
        
        user_lower = user_input.lower()
        return any(trigger in user_lower for trigger in triggers)
    
    def extract_repo_path(self, user_input: str) -> Optional[str]:
        """Extract repository path from user input"""
        import re
        
        # Look for common path patterns
        path_patterns = [
            r'/[a-zA-Z0-9_/.-]+',  # Unix-style paths
            r'[a-zA-Z]:[\\a-zA-Z0-9_\\.-]+',  # Windows paths
            r'\./[a-zA-Z0-9_/.-]+',  # Relative paths
            r'~/[a-zA-Z0-9_/.-]+',  # Home directory paths
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                if os.path.exists(match) and os.path.isdir(match):
                    return match
        
        # Default to current workspace if no path specified
        if os.path.exists('/workspace') and os.path.isdir('/workspace'):
            return '/workspace'
        
        return None
    
    def extract_vision(self, user_input: str, repo_path: str) -> str:
        """Extract vision description from user input"""
        # Remove the repo path from the input to get the vision
        vision = user_input
        
        # Remove common prefixes
        prefixes_to_remove = [
            f'take this repo {repo_path}',
            f'take the repo {repo_path}',
            f'refactor repo {repo_path}',
            f'modify repo {repo_path}',
            'take this repo',
            'refactor repo',
            'modify repo',
            'edit repo to',
            'transform repo',
            'align repo with'
        ]
        
        vision_lower = vision.lower()
        for prefix in prefixes_to_remove:
            if vision_lower.startswith(prefix.lower()):
                vision = vision[len(prefix):].strip()
                break
        
        # Remove leading conjunctions
        conjunctions = ['to', 'and', 'so that', 'according to']
        for conj in conjunctions:
            if vision.lower().startswith(conj):
                vision = vision[len(conj):].strip()
                break
        
        return vision if vision else "improve the codebase structure and functionality"
    
    async def process_refactor_request(self, user_input: str) -> Dict[str, Any]:
        """Process a repository refactoring request"""
        try:
            # Extract repo path and vision
            repo_path = self.extract_repo_path(user_input)
            if not repo_path:
                return {
                    'success': False,
                    'error': 'Could not find a valid repository path. Please specify the path to your repository.',
                    'suggestion': 'Try: "Take the repo /path/to/your/repo and convert it to microservices"'
                }
            
            vision = self.extract_vision(user_input, repo_path)
            
            print(f"🎯 Starting autonomous refactoring...")
            print(f"📁 Repository: {repo_path}")
            print(f"🔮 Vision: {vision}")
            print(f"{'='*60}")
            
            # Clear previous progress
            self.progress_log = []
            
            # Run refactoring in a separate thread to avoid blocking
            result = await self._run_refactoring_async(repo_path, vision)
            
            # Format response for OpenHands UI
            return self._format_response(result, repo_path, vision)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Refactoring failed: {str(e)}',
                'progress_log': self.progress_log
            }
    
    async def _run_refactoring_async(self, repo_path: str, vision: str) -> RefactorResult:
        """Run refactoring asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run the refactoring in a thread pool to avoid blocking
        result = await loop.run_in_executor(
            None, 
            self.wade_system.refactor_repository, 
            repo_path, 
            vision
        )
        
        return result
    
    def _format_response(self, result: RefactorResult, repo_path: str, vision: str) -> Dict[str, Any]:
        """Format refactoring result for OpenHands UI"""
        response = {
            'success': result.success,
            'repo_path': repo_path,
            'vision': vision,
            'summary': result.summary,
            'progress_log': self.progress_log,
            'changes': {
                'files_created': result.files_created,
                'files_modified': result.files_changed,
                'files_deleted': result.files_deleted
            },
            'test_results': result.test_results,
            'execution_output': result.execution_output,
            'errors': result.errors
        }
        
        # Add formatted output for display
        response['formatted_output'] = self._create_formatted_output(result)
        
        return response
    
    def _create_formatted_output(self, result: RefactorResult) -> str:
        """Create formatted output for display in OpenHands UI"""
        output_lines = []
        
        # Header
        output_lines.append("🤖 WADE Autonomous Refactoring Complete")
        output_lines.append("=" * 50)
        output_lines.append("")
        
        # Summary
        output_lines.append("📋 SUMMARY")
        output_lines.append("-" * 20)
        output_lines.append(result.summary)
        output_lines.append("")
        
        # Changes
        if result.files_created or result.files_changed or result.files_deleted:
            output_lines.append("📁 FILE CHANGES")
            output_lines.append("-" * 20)
            
            if result.files_created:
                output_lines.append(f"✨ Created ({len(result.files_created)}):")
                for file in result.files_created:
                    output_lines.append(f"   + {file}")
                output_lines.append("")
            
            if result.files_changed:
                output_lines.append(f"✏️ Modified ({len(result.files_changed)}):")
                for file in result.files_changed:
                    output_lines.append(f"   ~ {file}")
                output_lines.append("")
            
            if result.files_deleted:
                output_lines.append(f"🗑️ Deleted ({len(result.files_deleted)}):")
                for file in result.files_deleted:
                    output_lines.append(f"   - {file}")
                output_lines.append("")
        
        # Test Results
        if result.test_results:
            output_lines.append("🧪 TEST RESULTS")
            output_lines.append("-" * 20)
            
            if result.test_results.get('syntax_check'):
                syntax = result.test_results['syntax_check']
                output_lines.append(f"Syntax Check: {syntax['passed']} passed, {syntax['failed']} failed")
            
            if result.test_results.get('pytest'):
                pytest_result = result.test_results['pytest']
                if pytest_result.get('returncode') == 0:
                    output_lines.append("✅ All pytest tests passed")
                else:
                    output_lines.append("❌ Some pytest tests failed")
            
            output_lines.append("")
        
        # Execution Output
        if result.execution_output and result.execution_output != "":
            output_lines.append("🚀 EXECUTION RESULTS")
            output_lines.append("-" * 20)
            
            try:
                exec_data = eval(result.execution_output) if isinstance(result.execution_output, str) else result.execution_output
                if isinstance(exec_data, dict):
                    if 'error' in exec_data:
                        output_lines.append(f"❌ Error: {exec_data['error']}")
                    elif 'status_code' in exec_data:
                        output_lines.append(f"🌐 Web server running - Status: {exec_data['status_code']}")
                        if 'response' in exec_data:
                            output_lines.append(f"Response: {exec_data['response'][:200]}...")
                    elif 'stdout' in exec_data:
                        output_lines.append(f"Output: {exec_data['stdout'][:200]}...")
                else:
                    output_lines.append(str(exec_data)[:200] + "...")
            except:
                output_lines.append(str(result.execution_output)[:200] + "...")
            
            output_lines.append("")
        
        # Errors
        if result.errors:
            output_lines.append("❌ ERRORS")
            output_lines.append("-" * 20)
            for error in result.errors:
                output_lines.append(f"• {error}")
            output_lines.append("")
        
        # Next Steps
        output_lines.append("🎯 NEXT STEPS")
        output_lines.append("-" * 20)
        if result.success:
            output_lines.append("• Review the changes made to your repository")
            output_lines.append("• Test the refactored application manually")
            output_lines.append("• Commit changes to version control if satisfied")
            output_lines.append("• Deploy or continue development as needed")
        else:
            output_lines.append("• Review the errors above")
            output_lines.append("• Fix any issues manually if needed")
            output_lines.append("• Try refactoring again with a more specific vision")
        
        return "\n".join(output_lines)
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status"""
        return {
            'progress_log': self.progress_log,
            'current_step': len(self.progress_log),
            'is_running': self.current_task is not None
        }

# Global agent instance
refactor_agent = OpenHandsRefactorAgent()

async def handle_refactor_request(user_input: str) -> str:
    """Main handler for refactor requests - called by OpenHands"""
    
    # Check if this is a refactor request
    if not refactor_agent.detect_repo_refactor_intent(user_input):
        return "I don't detect a repository refactoring request. Try something like: 'Take this repo and convert it to microservices'"
    
    # Process the refactor request
    result = await refactor_agent.process_refactor_request(user_input)
    
    if result['success']:
        return result['formatted_output']
    else:
        error_msg = f"❌ Refactoring failed: {result['error']}\n\n"
        if result.get('progress_log'):
            error_msg += "Progress made:\n"
            for entry in result['progress_log']:
                error_msg += f"  [{entry['step']}/{entry['total_steps']}] {entry['message']}\n"
        return error_msg

def main():
    """Test function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python openhands_refactor_agent.py '<refactor request>'")
        print("Example: python openhands_refactor_agent.py 'Take this repo and convert it to FastAPI'")
        sys.exit(1)
    
    user_input = ' '.join(sys.argv[1:])
    
    async def test_run():
        result = await handle_refactor_request(user_input)
        print(result)
    
    asyncio.run(test_run())

if __name__ == "__main__":
    main()