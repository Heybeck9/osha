#!/usr/bin/env python3
"""
OpenHands UI Integration for WADE
Seamless integration with OpenHands frontend - no terminal needed
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class UIConfig:
    """Configuration for OpenHands UI integration"""
    vscode_port: int = 8080
    server_port: int = 12000
    enable_live_preview: bool = True
    auto_open_files: bool = True
    show_progress_panel: bool = True

class OpenHandsUIManager:
    """Manages UI integration with OpenHands frontend"""
    
    def __init__(self, config: UIConfig = None):
        self.config = config or UIConfig()
        self.active_sessions = {}
        self.ui_state = {
            'current_repo': None,
            'refactor_progress': [],
            'file_changes': [],
            'execution_results': {},
            'vscode_embedded': False
        }
    
    def create_ui_response(self, refactor_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a rich UI response for OpenHands frontend"""
        
        ui_response = {
            'type': 'wade_refactor_complete',
            'success': refactor_result['success'],
            'data': {
                'summary': refactor_result['summary'],
                'repo_path': refactor_result['repo_path'],
                'vision': refactor_result['vision'],
                'changes': refactor_result['changes'],
                'progress_log': refactor_result['progress_log']
            },
            'ui_components': self._generate_ui_components(refactor_result),
            'actions': self._generate_action_buttons(refactor_result)
        }
        
        return ui_response
    
    def _generate_ui_components(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI components for the frontend"""
        
        components = {
            'progress_panel': {
                'type': 'progress_summary',
                'title': '🤖 WADE Refactoring Progress',
                'items': result['progress_log'],
                'status': 'completed' if result['success'] else 'failed'
            },
            
            'changes_panel': {
                'type': 'file_changes',
                'title': '📁 File Changes',
                'created': result['changes']['files_created'],
                'modified': result['changes']['files_modified'], 
                'deleted': result['changes']['files_deleted']
            },
            
            'results_panel': {
                'type': 'execution_results',
                'title': '🚀 Execution Results',
                'test_results': result.get('test_results', {}),
                'execution_output': result.get('execution_output', ''),
                'success': result['success']
            },
            
            'code_preview': {
                'type': 'embedded_editor',
                'title': '💻 Code Preview',
                'files': self._get_preview_files(result),
                'vscode_url': f"http://localhost:{self.config.vscode_port}",
                'embedded': True
            }
        }
        
        return components
    
    def _generate_action_buttons(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action buttons for the UI"""
        
        actions = []
        
        if result['success']:
            actions.extend([
                {
                    'id': 'open_vscode',
                    'label': '📝 Open in VS Code',
                    'type': 'primary',
                    'action': 'embed_vscode',
                    'url': f"http://localhost:{self.config.vscode_port}?folder={result['repo_path']}"
                },
                {
                    'id': 'test_app',
                    'label': '🧪 Test Application',
                    'type': 'secondary', 
                    'action': 'run_tests',
                    'repo_path': result['repo_path']
                },
                {
                    'id': 'run_app',
                    'label': '🚀 Run Application',
                    'type': 'success',
                    'action': 'start_server',
                    'repo_path': result['repo_path']
                },
                {
                    'id': 'commit_changes',
                    'label': '💾 Commit Changes',
                    'type': 'info',
                    'action': 'git_commit',
                    'repo_path': result['repo_path']
                }
            ])
        else:
            actions.extend([
                {
                    'id': 'retry_refactor',
                    'label': '🔄 Retry Refactoring',
                    'type': 'warning',
                    'action': 'retry_refactor',
                    'repo_path': result['repo_path']
                },
                {
                    'id': 'debug_issues',
                    'label': '🐛 Debug Issues',
                    'type': 'danger',
                    'action': 'debug_mode',
                    'repo_path': result['repo_path']
                }
            ])
        
        return actions
    
    def _get_preview_files(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get files for code preview"""
        
        preview_files = []
        repo_path = Path(result['repo_path'])
        
        # Add changed files
        for file_path in result['changes']['files_modified'] + result['changes']['files_created']:
            full_path = repo_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    preview_files.append({
                        'path': file_path,
                        'content': content,
                        'language': self._detect_language(file_path),
                        'status': 'modified' if file_path in result['changes']['files_modified'] else 'created'
                    })
                except Exception as e:
                    preview_files.append({
                        'path': file_path,
                        'content': f"Error reading file: {e}",
                        'language': 'text',
                        'status': 'error'
                    })
        
        return preview_files[:10]  # Limit to 10 files for UI performance
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.sh': 'bash',
            '.dockerfile': 'dockerfile'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'text')
    
    def create_embedded_vscode_config(self, repo_path: str) -> Dict[str, Any]:
        """Create configuration for embedded VS Code"""
        
        return {
            'type': 'embedded_vscode',
            'config': {
                'url': f"http://localhost:{self.config.vscode_port}",
                'workspace': repo_path,
                'settings': {
                    'workbench.colorTheme': 'Default Dark+',
                    'editor.fontSize': 14,
                    'editor.tabSize': 4,
                    'files.autoSave': 'afterDelay',
                    'python.defaultInterpreterPath': '/usr/bin/python3',
                    'extensions.autoUpdate': False
                },
                'extensions': [
                    'ms-python.python',
                    'ms-python.flake8',
                    'ms-python.black-formatter',
                    'bradlc.vscode-tailwindcss'
                ]
            },
            'iframe_options': {
                'width': '100%',
                'height': '600px',
                'frameborder': '0',
                'allow': 'clipboard-read; clipboard-write'
            }
        }
    
    def create_live_server_panel(self, repo_path: str, port: int = 8000) -> Dict[str, Any]:
        """Create live server preview panel"""
        
        return {
            'type': 'live_server',
            'config': {
                'server_url': f"http://localhost:{port}",
                'repo_path': repo_path,
                'auto_reload': True,
                'endpoints': [
                    {'path': '/', 'method': 'GET', 'description': 'Home endpoint'},
                    {'path': '/docs', 'method': 'GET', 'description': 'API Documentation'},
                    {'path': '/users', 'method': 'GET', 'description': 'Get all users'},
                    {'path': '/users', 'method': 'POST', 'description': 'Create user'},
                    {'path': '/items', 'method': 'GET', 'description': 'Get all items'},
                    {'path': '/items', 'method': 'POST', 'description': 'Create item'}
                ]
            },
            'iframe_options': {
                'width': '100%',
                'height': '400px',
                'frameborder': '0'
            }
        }
    
    def generate_dashboard_layout(self, refactor_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete dashboard layout for OpenHands"""
        
        layout = {
            'type': 'wade_dashboard',
            'title': '🤖 WADE Autonomous Refactoring Dashboard',
            'layout': 'grid',
            'sections': [
                {
                    'id': 'summary',
                    'title': '📋 Summary',
                    'type': 'summary_card',
                    'content': refactor_result['summary'],
                    'status': 'success' if refactor_result['success'] else 'error',
                    'grid': {'row': 1, 'col': 1, 'span': 2}
                },
                {
                    'id': 'progress',
                    'title': '⏳ Progress',
                    'type': 'progress_timeline',
                    'content': refactor_result['progress_log'],
                    'grid': {'row': 1, 'col': 3, 'span': 1}
                },
                {
                    'id': 'code_editor',
                    'title': '💻 Code Editor',
                    'type': 'embedded_vscode',
                    'content': self.create_embedded_vscode_config(refactor_result['repo_path']),
                    'grid': {'row': 2, 'col': 1, 'span': 2}
                },
                {
                    'id': 'live_preview',
                    'title': '🚀 Live Preview',
                    'type': 'live_server',
                    'content': self.create_live_server_panel(refactor_result['repo_path']),
                    'grid': {'row': 2, 'col': 3, 'span': 1}
                },
                {
                    'id': 'file_changes',
                    'title': '📁 File Changes',
                    'type': 'file_tree',
                    'content': refactor_result['changes'],
                    'grid': {'row': 3, 'col': 1, 'span': 1}
                },
                {
                    'id': 'test_results',
                    'title': '🧪 Test Results',
                    'type': 'test_panel',
                    'content': refactor_result.get('test_results', {}),
                    'grid': {'row': 3, 'col': 2, 'span': 1}
                },
                {
                    'id': 'actions',
                    'title': '⚡ Actions',
                    'type': 'action_panel',
                    'content': self._generate_action_buttons(refactor_result),
                    'grid': {'row': 3, 'col': 3, 'span': 1}
                }
            ]
        }
        
        return layout

class WADEUIIntegration:
    """Main integration class for WADE with OpenHands UI"""
    
    def __init__(self):
        self.ui_manager = OpenHandsUIManager()
        self.active_sessions = {}
    
    async def process_refactor_with_ui(self, user_input: str) -> Dict[str, Any]:
        """Process refactor request and return rich UI response"""
        
        # Import the refactor agent
        from openhands_refactor_agent import handle_refactor_request
        
        # Process the refactor request
        refactor_result = await handle_refactor_request(user_input)
        
        # Convert to structured data if it's a string
        if isinstance(refactor_result, str):
            # Parse the text response into structured data
            structured_result = self._parse_text_response(refactor_result, user_input)
        else:
            structured_result = refactor_result
        
        # Generate rich UI response
        ui_response = self.ui_manager.create_ui_response(structured_result)
        
        # Generate dashboard layout
        dashboard = self.ui_manager.generate_dashboard_layout(structured_result)
        
        return {
            'text_response': refactor_result if isinstance(refactor_result, str) else structured_result.get('formatted_output', ''),
            'ui_response': ui_response,
            'dashboard': dashboard,
            'actions': ui_response['actions']
        }
    
    def _parse_text_response(self, text_response: str, user_input: str) -> Dict[str, Any]:
        """Parse text response into structured data"""
        
        # Extract repo path from user input
        repo_path = self._extract_repo_path(user_input)
        vision = self._extract_vision(user_input, repo_path)
        
        # Determine success from response text
        success = '✅' in text_response and 'completed' in text_response.lower()
        
        # Extract file changes (basic parsing)
        files_created = []
        files_modified = []
        files_deleted = []
        
        if 'Created' in text_response:
            # Basic extraction - could be improved with regex
            pass
        
        if 'Modified' in text_response:
            # Basic extraction - could be improved with regex
            pass
        
        return {
            'success': success,
            'repo_path': repo_path or '/workspace',
            'vision': vision,
            'summary': text_response[:200] + '...' if len(text_response) > 200 else text_response,
            'changes': {
                'files_created': files_created,
                'files_modified': files_modified,
                'files_deleted': files_deleted
            },
            'progress_log': [
                {'step': 1, 'total_steps': 7, 'message': 'Parsing vision and requirements...', 'timestamp': ''},
                {'step': 7, 'total_steps': 7, 'message': 'Generating summary...', 'timestamp': ''}
            ],
            'test_results': {},
            'execution_output': '',
            'formatted_output': text_response
        }
    
    def _extract_repo_path(self, user_input: str) -> Optional[str]:
        """Extract repository path from user input"""
        import re
        
        # Look for path patterns
        path_patterns = [
            r'/[a-zA-Z0-9_/.-]+',
            r'\./[a-zA-Z0-9_/.-]+',
            r'~/[a-zA-Z0-9_/.-]+'
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                if os.path.exists(match) and os.path.isdir(match):
                    return match
        
        return None
    
    def _extract_vision(self, user_input: str, repo_path: str) -> str:
        """Extract vision from user input"""
        vision = user_input.lower()
        
        # Remove repo path references
        if repo_path:
            vision = vision.replace(repo_path.lower(), '')
        
        # Remove common prefixes
        prefixes = ['take this repo', 'take the repo', 'refactor', 'convert', 'transform']
        for prefix in prefixes:
            if vision.startswith(prefix):
                vision = vision[len(prefix):].strip()
                break
        
        return vision or 'improve the codebase'

# Global integration instance
wade_ui = WADEUIIntegration()

async def handle_refactor_with_ui(user_input: str) -> Dict[str, Any]:
    """Main handler for UI-integrated refactor requests"""
    return await wade_ui.process_refactor_with_ui(user_input)

def main():
    """Test the UI integration"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python openhands_ui_integration.py '<refactor request>'")
        sys.exit(1)
    
    user_input = ' '.join(sys.argv[1:])
    
    async def test_run():
        result = await handle_refactor_with_ui(user_input)
        print("=== UI Integration Result ===")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_run())

if __name__ == "__main__":
    main()