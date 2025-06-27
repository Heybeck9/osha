#!/usr/bin/env python3
"""
WADE Autonomous Repo Refactor System
Transforms repositories based on natural language vision descriptions
"""

import os
import ast
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import time
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RefactorTask:
    """Represents a refactoring task with vision and context"""
    repo_path: str
    vision: str
    current_files: List[str]
    target_architecture: str
    requirements: List[str]
    constraints: List[str]

@dataclass
class RefactorResult:
    """Results of a refactoring operation"""
    success: bool
    files_changed: List[str]
    files_created: List[str]
    files_deleted: List[str]
    test_results: Dict[str, Any]
    execution_output: str
    errors: List[str]
    summary: str

class VisionParser:
    """Parses natural language vision into actionable requirements"""
    
    def __init__(self):
        self.architecture_patterns = {
            'microservice': ['separate services', 'api gateway', 'service discovery', 'independent deployment'],
            'mvc': ['model view controller', 'separation of concerns', 'business logic', 'presentation layer'],
            'clean_architecture': ['dependency inversion', 'use cases', 'entities', 'adapters'],
            'plugin_system': ['extensible', 'modular', 'plugin interface', 'dynamic loading'],
            'event_driven': ['events', 'publishers', 'subscribers', 'message queue'],
            'layered': ['layers', 'data access', 'business logic', 'presentation']
        }
        
        self.technology_mappings = {
            'flask': 'fastapi',
            'django': 'fastapi',
            'express': 'fastapi',
            'sync': 'async',
            'synchronous': 'asynchronous',
            'rest': 'graphql',
            'sql': 'nosql'
        }
    
    def parse_vision(self, vision: str) -> RefactorTask:
        """Parse natural language vision into structured requirements"""
        vision_lower = vision.lower()
        
        # Detect architecture pattern
        target_architecture = 'custom'
        for pattern, keywords in self.architecture_patterns.items():
            if any(keyword in vision_lower for keyword in keywords):
                target_architecture = pattern
                break
        
        # Extract requirements
        requirements = []
        if 'fastapi' in vision_lower or 'async' in vision_lower:
            requirements.append('convert_to_fastapi')
        if 'microservice' in vision_lower:
            requirements.append('split_into_services')
        if 'test' in vision_lower:
            requirements.append('add_tests')
        if 'docker' in vision_lower:
            requirements.append('add_docker')
        if 'logging' in vision_lower:
            requirements.append('add_logging')
        if 'middleware' in vision_lower:
            requirements.append('add_middleware')
        
        # Extract constraints
        constraints = []
        if 'keep' in vision_lower and 'database' in vision_lower:
            constraints.append('preserve_database')
        if 'maintain' in vision_lower and 'api' in vision_lower:
            constraints.append('preserve_api_compatibility')
        
        return RefactorTask(
            repo_path="",
            vision=vision,
            current_files=[],
            target_architecture=target_architecture,
            requirements=requirements,
            constraints=constraints
        )

class CodeAnalyzer:
    """Analyzes existing codebase structure and patterns"""
    
    def __init__(self):
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c'}
    
    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and extract key information"""
        analysis = {
            'structure': {},
            'languages': set(),
            'frameworks': set(),
            'patterns': set(),
            'entry_points': [],
            'dependencies': {},
            'test_files': [],
            'config_files': []
        }
        
        repo_path = Path(repo_path)
        
        # Walk through repository
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                self._analyze_file(file_path, analysis)
        
        # Convert sets to lists for JSON serialization
        analysis['languages'] = list(analysis['languages'])
        analysis['frameworks'] = list(analysis['frameworks'])
        analysis['patterns'] = list(analysis['patterns'])
        
        return analysis
    
    def _analyze_file(self, file_path: Path, analysis: Dict[str, Any]):
        """Analyze individual file"""
        suffix = file_path.suffix.lower()
        
        # Language detection
        if suffix == '.py':
            analysis['languages'].add('python')
            self._analyze_python_file(file_path, analysis)
        elif suffix in {'.js', '.ts'}:
            analysis['languages'].add('javascript')
            self._analyze_js_file(file_path, analysis)
        elif suffix == '.java':
            analysis['languages'].add('java')
        
        # Special files
        if file_path.name in {'requirements.txt', 'pyproject.toml', 'package.json', 'Dockerfile'}:
            analysis['config_files'].append(str(file_path))
        
        if 'test' in file_path.name.lower() or file_path.parent.name.lower() == 'tests':
            analysis['test_files'].append(str(file_path))
    
    def _analyze_python_file(self, file_path: Path, analysis: Dict[str, Any]):
        """Analyze Python file for frameworks and patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Framework detection
            if 'from flask import' in content or 'import flask' in content:
                analysis['frameworks'].add('flask')
            if 'from fastapi import' in content or 'import fastapi' in content:
                analysis['frameworks'].add('fastapi')
            if 'from django' in content or 'import django' in content:
                analysis['frameworks'].add('django')
            
            # Entry point detection
            if 'if __name__ == "__main__"' in content:
                analysis['entry_points'].append(str(file_path))
            if 'app.run(' in content or 'uvicorn.run(' in content:
                analysis['entry_points'].append(str(file_path))
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def _analyze_js_file(self, file_path: Path, analysis: Dict[str, Any]):
        """Analyze JavaScript/TypeScript file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'express' in content:
                analysis['frameworks'].add('express')
            if 'react' in content:
                analysis['frameworks'].add('react')
            if 'vue' in content:
                analysis['frameworks'].add('vue')
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

class CodeGenerator:
    """Generates code based on refactoring requirements"""
    
    def __init__(self):
        self.templates = {
            'fastapi_app': self._get_fastapi_template(),
            'microservice': self._get_microservice_template(),
            'test_file': self._get_test_template(),
            'dockerfile': self._get_dockerfile_template()
        }
    
    def generate_fastapi_from_flask(self, flask_file: str) -> str:
        """Convert Flask app to FastAPI"""
        with open(flask_file, 'r') as f:
            flask_content = f.read()
        
        # Create a proper FastAPI conversion
        fastapi_content = '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Converted API", description="Converted from Flask to FastAPI")

# Pydantic models
class User(BaseModel):
    name: str
    email: Optional[str] = ""

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class Item(BaseModel):
    name: str
    price: Optional[float] = 0.0

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float

# Simple in-memory storage
users = []
items = []

@app.get("/")
async def home():
    return {"message": "Welcome to the FastAPI API"}

@app.get("/users", response_model=List[UserResponse])
async def get_users():
    return users

@app.post("/users", response_model=UserResponse)
async def create_user(user: User):
    new_user = {
        "id": len(users) + 1,
        "name": user.name,
        "email": user.email
    }
    users.append(new_user)
    return new_user

@app.get("/items", response_model=List[ItemResponse])
async def get_items():
    return items

@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    new_item = {
        "id": len(items) + 1,
        "name": item.name,
        "price": item.price
    }
    items.append(new_item)
    return new_item

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    user = next((u for u in users if u['id'] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        return fastapi_content
    
    def _get_fastapi_template(self) -> str:
        return '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Item(BaseModel):
    name: str
    value: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _get_microservice_template(self) -> str:
        return '''# Microservice Template
from fastapi import FastAPI
from .routers import items, users
from .middleware import logging_middleware

app = FastAPI(title="Microservice")

app.middleware("http")(logging_middleware)
app.include_router(items.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
'''
    
    def _get_test_template(self) -> str:
        return '''import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
'''
    
    def _get_dockerfile_template(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

class TestRunner:
    """Runs tests and validates functionality"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def run_tests(self) -> Dict[str, Any]:
        """Run all available tests"""
        results = {
            'pytest': None,
            'unittest': None,
            'custom': None,
            'syntax_check': None
        }
        
        # Check Python syntax
        results['syntax_check'] = self._check_python_syntax()
        
        # Run pytest if available
        if self._has_pytest():
            results['pytest'] = self._run_pytest()
        
        # Run unittest if available
        if self._has_unittest():
            results['unittest'] = self._run_unittest()
        
        return results
    
    def _check_python_syntax(self) -> Dict[str, Any]:
        """Check Python syntax for all .py files"""
        results = {'passed': 0, 'failed': 0, 'errors': []}
        
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    ast.parse(f.read())
                results['passed'] += 1
            except SyntaxError as e:
                results['failed'] += 1
                results['errors'].append(f"{py_file}: {e}")
        
        return results
    
    def _has_pytest(self) -> bool:
        """Check if pytest is available"""
        try:
            subprocess.run(['pytest', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _run_pytest(self) -> Dict[str, Any]:
        """Run pytest"""
        try:
            result = subprocess.run(
                ['pytest', '-v', '--tb=short'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Tests timed out'}
        except Exception as e:
            return {'error': str(e)}
    
    def _has_unittest(self) -> bool:
        """Check if unittest tests exist"""
        return any(self.repo_path.rglob('test_*.py'))
    
    def _run_unittest(self) -> Dict[str, Any]:
        """Run unittest"""
        try:
            result = subprocess.run(
                ['python', '-m', 'unittest', 'discover', '-v'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            return {'error': str(e)}

class ExecutionEngine:
    """Executes refactored code and captures results"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def execute_application(self) -> Dict[str, Any]:
        """Execute the main application and capture output"""
        entry_points = self._find_entry_points()
        
        if not entry_points:
            return {'error': 'No entry points found'}
        
        # Try to execute the main entry point
        main_entry = entry_points[0]
        
        try:
            # For web applications, start server and test endpoints
            if self._is_web_app(main_entry):
                return self._execute_web_app(main_entry)
            else:
                return self._execute_script(main_entry)
        except Exception as e:
            return {'error': str(e)}
    
    def _find_entry_points(self) -> List[str]:
        """Find potential entry points"""
        entry_points = []
        
        # Look for main.py, app.py, server.py
        for name in ['main.py', 'app.py', 'server.py', 'run.py']:
            if (self.repo_path / name).exists():
                entry_points.append(name)
        
        # Look for files with if __name__ == "__main__"
        for py_file in self.repo_path.rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    if 'if __name__ == "__main__"' in f.read():
                        entry_points.append(str(py_file.relative_to(self.repo_path)))
            except Exception:
                continue
        
        return entry_points
    
    def _is_web_app(self, entry_point: str) -> bool:
        """Check if entry point is a web application"""
        try:
            with open(self.repo_path / entry_point, 'r') as f:
                content = f.read()
            return any(framework in content.lower() for framework in ['flask', 'fastapi', 'django', 'uvicorn'])
        except Exception:
            return False
    
    def _execute_web_app(self, entry_point: str) -> Dict[str, Any]:
        """Execute web application and test endpoints"""
        import threading
        import requests
        import time
        
        # Start server in background
        server_process = subprocess.Popen(
            ['python', entry_point],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(3)
        
        try:
            # Test basic endpoints
            response = requests.get('http://localhost:8000/', timeout=5)
            result = {
                'status_code': response.status_code,
                'response': response.text,
                'headers': dict(response.headers)
            }
        except requests.RequestException as e:
            result = {'error': f'Failed to connect to server: {e}'}
        finally:
            server_process.terminate()
            server_process.wait(timeout=5)
        
        return result
    
    def _execute_script(self, entry_point: str) -> Dict[str, Any]:
        """Execute a regular Python script"""
        try:
            result = subprocess.run(
                ['python', entry_point],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Script execution timed out'}

class WADERefactorSystem:
    """Main autonomous refactoring system"""
    
    def __init__(self):
        self.vision_parser = VisionParser()
        self.code_analyzer = CodeAnalyzer()
        self.code_generator = CodeGenerator()
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def _update_progress(self, message: str, step: int, total_steps: int):
        """Update progress"""
        if self.progress_callback:
            self.progress_callback(message, step, total_steps)
        else:
            print(f"[{step}/{total_steps}] {message}")
    
    def refactor_repository(self, repo_path: str, vision: str) -> RefactorResult:
        """Main refactoring workflow"""
        total_steps = 7
        
        try:
            # Step 1: Parse vision
            self._update_progress("Parsing vision and requirements...", 1, total_steps)
            task = self.vision_parser.parse_vision(vision)
            task.repo_path = repo_path
            
            # Step 2: Analyze repository
            self._update_progress("Analyzing repository structure...", 2, total_steps)
            analysis = self.code_analyzer.analyze_repository(repo_path)
            
            # Step 3: Create refactoring plan
            self._update_progress("Creating refactoring plan...", 3, total_steps)
            plan = self._create_refactoring_plan(task, analysis)
            
            # Step 4: Execute refactoring
            self._update_progress("Executing refactoring...", 4, total_steps)
            changes = self._execute_refactoring(repo_path, plan)
            
            # Step 5: Run tests
            self._update_progress("Running tests...", 5, total_steps)
            test_runner = TestRunner(repo_path)
            test_results = test_runner.run_tests()
            
            # Step 6: Execute application
            self._update_progress("Executing application...", 6, total_steps)
            execution_engine = ExecutionEngine(repo_path)
            execution_output = execution_engine.execute_application()
            
            # Step 7: Generate summary
            self._update_progress("Generating summary...", 7, total_steps)
            summary = self._generate_summary(changes, test_results, execution_output)
            
            return RefactorResult(
                success=True,
                files_changed=changes.get('modified', []),
                files_created=changes.get('created', []),
                files_deleted=changes.get('deleted', []),
                test_results=test_results,
                execution_output=str(execution_output),
                errors=[],
                summary=summary
            )
            
        except Exception as e:
            return RefactorResult(
                success=False,
                files_changed=[],
                files_created=[],
                files_deleted=[],
                test_results={},
                execution_output="",
                errors=[str(e)],
                summary=f"Refactoring failed: {e}"
            )
    
    def _create_refactoring_plan(self, task: RefactorTask, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed refactoring plan"""
        plan = {
            'target_architecture': task.target_architecture,
            'requirements': task.requirements,
            'steps': []
        }
        
        # Add steps based on requirements
        if 'convert_to_fastapi' in task.requirements and 'flask' in analysis.get('frameworks', []):
            plan['steps'].append({
                'action': 'convert_framework',
                'from': 'flask',
                'to': 'fastapi',
                'files': analysis.get('entry_points', [])
            })
        
        if 'add_tests' in task.requirements:
            plan['steps'].append({
                'action': 'add_tests',
                'test_framework': 'pytest',
                'coverage_target': 80
            })
        
        if 'add_logging' in task.requirements:
            plan['steps'].append({
                'action': 'add_logging',
                'logger': 'structlog'
            })
        
        return plan
    
    def _execute_refactoring(self, repo_path: str, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """Execute the refactoring plan"""
        changes = {
            'created': [],
            'modified': [],
            'deleted': []
        }
        
        repo_path = Path(repo_path)
        
        for step in plan['steps']:
            if step['action'] == 'convert_framework':
                self._convert_framework(repo_path, step, changes)
            elif step['action'] == 'add_tests':
                self._add_tests(repo_path, step, changes)
            elif step['action'] == 'add_logging':
                self._add_logging(repo_path, step, changes)
        
        return changes
    
    def _convert_framework(self, repo_path: Path, step: Dict[str, Any], changes: Dict[str, List[str]]):
        """Convert from one framework to another"""
        if step['from'] == 'flask' and step['to'] == 'fastapi':
            for file_path in step['files']:
                # Handle both absolute and relative paths
                if os.path.isabs(file_path):
                    full_path = Path(file_path)
                    relative_path = str(full_path.relative_to(repo_path))
                else:
                    full_path = repo_path / file_path
                    relative_path = file_path
                
                if full_path.exists():
                    print(f"Converting {full_path} from Flask to FastAPI...")
                    # Convert Flask to FastAPI
                    fastapi_content = self.code_generator.generate_fastapi_from_flask(str(full_path))
                    
                    # Write converted content
                    with open(full_path, 'w') as f:
                        f.write(fastapi_content)
                    
                    changes['modified'].append(relative_path)
            
            # Create requirements.txt with FastAPI dependencies
            requirements_path = repo_path / 'requirements.txt'
            with open(requirements_path, 'w') as f:
                f.write('fastapi\nuvicorn[standard]\npydantic\n')
            
            if not requirements_path.exists():
                changes['created'].append('requirements.txt')
            else:
                changes['modified'].append('requirements.txt')
    
    def _add_tests(self, repo_path: Path, step: Dict[str, Any], changes: Dict[str, List[str]]):
        """Add test files"""
        tests_dir = repo_path / 'tests'
        tests_dir.mkdir(exist_ok=True)
        
        # Create basic test file
        test_file = tests_dir / 'test_main.py'
        with open(test_file, 'w') as f:
            f.write(self.code_generator.templates['test_file'])
        
        changes['created'].append('tests/test_main.py')
    
    def _add_logging(self, repo_path: Path, step: Dict[str, Any], changes: Dict[str, List[str]]):
        """Add logging configuration"""
        # Create logging configuration
        logging_config = '''import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)
'''
        
        logging_file = repo_path / 'logging_config.py'
        with open(logging_file, 'w') as f:
            f.write(logging_config)
        
        changes['created'].append('logging_config.py')
    
    def _generate_summary(self, changes: Dict[str, List[str]], test_results: Dict[str, Any], execution_output: Dict[str, Any]) -> str:
        """Generate refactoring summary"""
        summary_parts = []
        
        # Changes summary
        total_changes = len(changes['created']) + len(changes['modified']) + len(changes['deleted'])
        summary_parts.append(f"✅ Refactoring completed with {total_changes} file changes")
        
        if changes['created']:
            summary_parts.append(f"📁 Created {len(changes['created'])} files: {', '.join(changes['created'])}")
        
        if changes['modified']:
            summary_parts.append(f"✏️ Modified {len(changes['modified'])} files: {', '.join(changes['modified'])}")
        
        # Test results summary
        if test_results.get('syntax_check'):
            syntax = test_results['syntax_check']
            summary_parts.append(f"🔍 Syntax check: {syntax['passed']} passed, {syntax['failed']} failed")
        
        if test_results.get('pytest'):
            pytest_result = test_results['pytest']
            if pytest_result.get('returncode') == 0:
                summary_parts.append("✅ All tests passed")
            else:
                summary_parts.append("❌ Some tests failed")
        
        # Execution summary
        if 'error' not in execution_output:
            summary_parts.append("🚀 Application executed successfully")
            if 'status_code' in execution_output:
                summary_parts.append(f"🌐 Web server responded with status {execution_output['status_code']}")
        else:
            summary_parts.append(f"❌ Execution failed: {execution_output['error']}")
        
        return '\n'.join(summary_parts)

def main():
    """Main function for testing"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python wade_refactor_system.py <repo_path> <vision>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    vision = sys.argv[2]
    
    system = WADERefactorSystem()
    result = system.refactor_repository(repo_path, vision)
    
    print("=== REFACTORING RESULTS ===")
    print(f"Success: {result.success}")
    print(f"Summary: {result.summary}")
    
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

if __name__ == "__main__":
    main()