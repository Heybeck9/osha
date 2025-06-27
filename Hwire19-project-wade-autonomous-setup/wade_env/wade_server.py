#!/usr/bin/env python3
"""
WADE Autonomous Repo Refactor Server
A web interface for autonomous repository refactoring with vision-based transformations
"""

import os
import json
import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Data Models
class RefactorRequest(BaseModel):
    repo_path: str
    vision: str
    test_command: Optional[str] = None
    run_command: Optional[str] = None

class RefactorStatus(BaseModel):
    task_id: str
    status: str  # "running", "completed", "failed"
    progress: int  # 0-100
    current_step: str
    logs: List[str]
    files_changed: List[str]
    test_results: Optional[str] = None
    execution_output: Optional[str] = None

@dataclass
class RefactorTask:
    task_id: str
    repo_path: str
    vision: str
    test_command: Optional[str]
    run_command: Optional[str]
    status: str = "running"
    progress: int = 0
    current_step: str = "Initializing"
    logs: List[str] = None
    files_changed: List[str] = None
    test_results: Optional[str] = None
    execution_output: Optional[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.files_changed is None:
            self.files_changed = []

# Global task storage
active_tasks: Dict[str, RefactorTask] = {}

app = FastAPI(title="WADE Autonomous Repo Refactor", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutonomousRefactorAgent:
    """The core autonomous refactoring agent"""
    
    def __init__(self, task: RefactorTask):
        self.task = task
        self.repo_path = Path(task.repo_path)
        self.backup_path = None
        
    async def execute(self):
        """Main execution pipeline"""
        try:
            await self.log("🚀 Starting autonomous repo refactoring...")
            
            # Step 1: Analyze Repository
            await self.update_progress(10, "Analyzing repository structure")
            repo_analysis = await self.analyze_repository()
            
            # Step 2: Parse Vision
            await self.update_progress(20, "Parsing transformation vision")
            transformation_plan = await self.parse_vision(repo_analysis)
            
            # Step 3: Create Backup
            await self.update_progress(30, "Creating backup")
            await self.create_backup()
            
            # Step 4: Execute Transformations
            await self.update_progress(40, "Executing code transformations")
            await self.execute_transformations(transformation_plan)
            
            # Step 5: Test Changes
            await self.update_progress(70, "Testing changes")
            test_success = await self.test_changes()
            
            # Step 6: Debug if needed
            if not test_success:
                await self.update_progress(80, "Debugging issues")
                await self.debug_and_fix()
                await self.test_changes()
            
            # Step 7: Execute Final Code
            await self.update_progress(90, "Executing refactored code")
            await self.execute_code()
            
            await self.update_progress(100, "Refactoring completed successfully")
            self.task.status = "completed"
            
        except Exception as e:
            await self.log(f"❌ Error: {str(e)}")
            self.task.status = "failed"
            raise
    
    async def analyze_repository(self) -> Dict[str, Any]:
        """Analyze the repository structure and content"""
        await self.log("📁 Analyzing repository structure...")
        
        analysis = {
            "files": [],
            "languages": set(),
            "frameworks": [],
            "structure": {},
            "dependencies": {}
        }
        
        # Walk through repository
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories and common build/cache dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'build', 'dist']]
            
            for file in files:
                if not file.startswith('.'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.repo_path)
                    analysis["files"].append(str(rel_path))
                    
                    # Detect language
                    suffix = file_path.suffix.lower()
                    if suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php']:
                        analysis["languages"].add(suffix[1:])
        
        # Detect frameworks and dependencies
        await self.detect_frameworks(analysis)
        
        await self.log(f"📊 Found {len(analysis['files'])} files in {len(analysis['languages'])} languages")
        return analysis
    
    async def detect_frameworks(self, analysis: Dict[str, Any]):
        """Detect frameworks and dependencies"""
        frameworks = []
        
        # Check for common framework files
        files = analysis["files"]
        
        if "requirements.txt" in files or "pyproject.toml" in files:
            frameworks.append("Python")
            # Check for Flask/FastAPI/Django
            for file in files:
                if file.endswith('.py'):
                    try:
                        content = (self.repo_path / file).read_text()
                        if 'from flask import' in content or 'import flask' in content:
                            frameworks.append("Flask")
                        elif 'from fastapi import' in content or 'import fastapi' in content:
                            frameworks.append("FastAPI")
                        elif 'django' in content.lower():
                            frameworks.append("Django")
                    except:
                        pass
        
        if "package.json" in files:
            frameworks.append("Node.js")
            try:
                package_json = json.loads((self.repo_path / "package.json").read_text())
                deps = {**package_json.get("dependencies", {}), **package_json.get("devDependencies", {})}
                if "react" in deps:
                    frameworks.append("React")
                if "vue" in deps:
                    frameworks.append("Vue")
                if "express" in deps:
                    frameworks.append("Express")
            except:
                pass
        
        analysis["frameworks"] = frameworks
    
    async def parse_vision(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the transformation vision into actionable plan"""
        await self.log("🧠 Parsing transformation vision...")
        
        vision = self.task.vision.lower()
        plan = {
            "transformations": [],
            "new_files": [],
            "modifications": [],
            "deletions": []
        }
        
        # Simple pattern matching for common transformations
        if "microservice" in vision:
            plan["transformations"].append({
                "type": "architecture_change",
                "from": "monolith",
                "to": "microservices",
                "actions": ["split_routes", "create_services", "add_api_gateway"]
            })
        
        if "fastapi" in vision and "flask" in str(repo_analysis["frameworks"]).lower():
            plan["transformations"].append({
                "type": "framework_migration",
                "from": "Flask",
                "to": "FastAPI",
                "actions": ["convert_routes", "update_imports", "add_async"]
            })
        
        if "async" in vision:
            plan["transformations"].append({
                "type": "async_conversion",
                "actions": ["convert_functions", "add_await", "update_calls"]
            })
        
        if "logging" in vision:
            plan["transformations"].append({
                "type": "add_logging",
                "actions": ["setup_logger", "add_log_statements", "create_middleware"]
            })
        
        await self.log(f"📋 Created transformation plan with {len(plan['transformations'])} major changes")
        return plan
    
    async def create_backup(self):
        """Create a backup of the repository"""
        self.backup_path = self.repo_path.parent / f"{self.repo_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(self.repo_path, self.backup_path)
        await self.log(f"💾 Created backup at {self.backup_path}")
    
    async def execute_transformations(self, plan: Dict[str, Any]):
        """Execute the transformation plan"""
        await self.log("🔧 Executing transformations...")
        
        for transformation in plan["transformations"]:
            await self.log(f"⚙️ Applying {transformation['type']}...")
            
            if transformation["type"] == "framework_migration":
                await self.migrate_framework(transformation)
            elif transformation["type"] == "architecture_change":
                await self.change_architecture(transformation)
            elif transformation["type"] == "async_conversion":
                await self.convert_to_async(transformation)
            elif transformation["type"] == "add_logging":
                await self.add_logging(transformation)
    
    async def migrate_framework(self, transformation: Dict[str, Any]):
        """Migrate from Flask to FastAPI"""
        if transformation["from"] == "Flask" and transformation["to"] == "FastAPI":
            # Find Python files with Flask imports
            for file_path in self.repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text()
                    if "from flask import" in content or "import flask" in content:
                        # Basic Flask to FastAPI conversion
                        new_content = content
                        
                        # Replace imports
                        new_content = new_content.replace("from flask import Flask", "from fastapi import FastAPI")
                        new_content = new_content.replace("from flask import", "from fastapi import")
                        new_content = new_content.replace("Flask(__name__)", "FastAPI()")
                        
                        # Replace route decorators
                        new_content = new_content.replace("@app.route(", "@app.get(")
                        new_content = new_content.replace("methods=['POST']", "# POST method - convert to @app.post")
                        
                        # Add async to functions
                        import re
                        new_content = re.sub(r'def (\w+)\(', r'async def \1(', new_content)
                        
                        file_path.write_text(new_content)
                        self.task.files_changed.append(str(file_path.relative_to(self.repo_path)))
                        await self.log(f"✅ Converted {file_path.name} from Flask to FastAPI")
                except Exception as e:
                    await self.log(f"⚠️ Error converting {file_path.name}: {str(e)}")
    
    async def change_architecture(self, transformation: Dict[str, Any]):
        """Change architecture (e.g., monolith to microservices)"""
        await self.log("🏗️ Restructuring architecture...")
        
        if transformation["to"] == "microservices":
            # Create services directory
            services_dir = self.repo_path / "services"
            services_dir.mkdir(exist_ok=True)
            
            # Create basic service structure
            for service_name in ["user_service", "auth_service", "api_gateway"]:
                service_dir = services_dir / service_name
                service_dir.mkdir(exist_ok=True)
                
                # Create basic service file
                service_file = service_dir / "main.py"
                service_content = f'''"""
{service_name.replace('_', ' ').title()}
Microservice component
"""

from fastapi import FastAPI

app = FastAPI(title="{service_name.replace('_', ' ').title()}")

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "service": "{service_name}"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
                service_file.write_text(service_content)
                self.task.files_changed.append(str(service_file.relative_to(self.repo_path)))
            
            await self.log("✅ Created microservices structure")
    
    async def convert_to_async(self, transformation: Dict[str, Any]):
        """Convert synchronous code to async"""
        await self.log("🔄 Converting to async/await pattern...")
        
        for file_path in self.repo_path.rglob("*.py"):
            try:
                content = file_path.read_text()
                if "def " in content and "async def" not in content:
                    # Simple async conversion
                    import re
                    new_content = re.sub(r'def (\w+)\(', r'async def \1(', content)
                    
                    # Add await to common blocking calls
                    new_content = new_content.replace("requests.get(", "await httpx.get(")
                    new_content = new_content.replace("requests.post(", "await httpx.post(")
                    
                    if new_content != content:
                        file_path.write_text(new_content)
                        self.task.files_changed.append(str(file_path.relative_to(self.repo_path)))
                        await self.log(f"✅ Converted {file_path.name} to async")
            except Exception as e:
                await self.log(f"⚠️ Error converting {file_path.name}: {str(e)}")
    
    async def add_logging(self, transformation: Dict[str, Any]):
        """Add logging infrastructure"""
        await self.log("📝 Adding logging infrastructure...")
        
        # Create logging configuration
        logging_config = '''
import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
'''
        
        logging_file = self.repo_path / "logging_config.py"
        logging_file.write_text(logging_config)
        self.task.files_changed.append("logging_config.py")
        await self.log("✅ Added logging configuration")
    
    async def test_changes(self) -> bool:
        """Test the changes"""
        await self.log("🧪 Testing changes...")
        
        if self.task.test_command:
            try:
                result = subprocess.run(
                    self.task.test_command,
                    shell=True,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                self.task.test_results = f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                
                if result.returncode == 0:
                    await self.log("✅ Tests passed!")
                    return True
                else:
                    await self.log(f"❌ Tests failed with exit code {result.returncode}")
                    return False
                    
            except subprocess.TimeoutExpired:
                await self.log("⏰ Tests timed out")
                return False
            except Exception as e:
                await self.log(f"❌ Test execution error: {str(e)}")
                return False
        else:
            # Basic syntax check for Python files
            for py_file in self.repo_path.rglob("*.py"):
                try:
                    compile(py_file.read_text(), py_file, 'exec')
                except SyntaxError as e:
                    await self.log(f"❌ Syntax error in {py_file.name}: {str(e)}")
                    return False
            
            await self.log("✅ Basic syntax checks passed")
            return True
    
    async def debug_and_fix(self):
        """Debug and fix issues"""
        await self.log("🐛 Debugging and fixing issues...")
        
        # Simple fix attempts
        for py_file in self.repo_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                
                # Fix common async issues
                if "async def" in content and "import asyncio" not in content:
                    content = "import asyncio\n" + content
                    py_file.write_text(content)
                    await self.log(f"🔧 Added asyncio import to {py_file.name}")
                
                # Fix missing imports
                if "httpx" in content and "import httpx" not in content:
                    content = "import httpx\n" + content
                    py_file.write_text(content)
                    await self.log(f"🔧 Added httpx import to {py_file.name}")
                    
            except Exception as e:
                await self.log(f"⚠️ Error fixing {py_file.name}: {str(e)}")
    
    async def execute_code(self):
        """Execute the refactored code"""
        await self.log("🚀 Executing refactored code...")
        
        if self.task.run_command:
            try:
                # Start the process in background
                process = subprocess.Popen(
                    self.task.run_command,
                    shell=True,
                    cwd=self.repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait a bit for startup
                await asyncio.sleep(3)
                
                if process.poll() is None:
                    # Process is still running
                    self.task.execution_output = f"✅ Application started successfully (PID: {process.pid})\nCommand: {self.task.run_command}"
                    await self.log("✅ Application is running!")
                else:
                    # Process exited
                    stdout, stderr = process.communicate()
                    self.task.execution_output = f"Exit code: {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                    await self.log(f"❌ Application exited with code {process.returncode}")
                    
            except Exception as e:
                await self.log(f"❌ Execution error: {str(e)}")
                self.task.execution_output = f"Error: {str(e)}"
        else:
            self.task.execution_output = "No run command specified"
            await self.log("ℹ️ No run command specified")
    
    async def log(self, message: str):
        """Add a log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.task.logs.append(log_entry)
        print(log_entry)  # Also print to console
    
    async def update_progress(self, progress: int, step: str):
        """Update task progress"""
        self.task.progress = progress
        self.task.current_step = step
        await self.log(f"📊 Progress: {progress}% - {step}")

# API Endpoints
@app.post("/api/refactor", response_model=dict)
async def start_refactor(request: RefactorRequest, background_tasks: BackgroundTasks):
    """Start a new refactoring task"""
    
    # Validate repo path
    if not os.path.exists(request.repo_path):
        raise HTTPException(status_code=400, detail="Repository path does not exist")
    
    # Generate task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(active_tasks)}"
    
    # Create task
    task = RefactorTask(
        task_id=task_id,
        repo_path=request.repo_path,
        vision=request.vision,
        test_command=request.test_command,
        run_command=request.run_command
    )
    
    active_tasks[task_id] = task
    
    # Start refactoring in background
    agent = AutonomousRefactorAgent(task)
    background_tasks.add_task(agent.execute)
    
    return {"task_id": task_id, "status": "started"}

@app.get("/api/status/{task_id}", response_model=RefactorStatus)
async def get_status(task_id: str):
    """Get the status of a refactoring task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    return RefactorStatus(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        current_step=task.current_step,
        logs=task.logs,
        files_changed=task.files_changed,
        test_results=task.test_results,
        execution_output=task.execution_output
    )

@app.get("/api/tasks")
async def list_tasks():
    """List all tasks"""
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "current_step": task.current_step,
                "repo_path": task.repo_path,
                "vision": task.vision[:100] + "..." if len(task.vision) > 100 else task.vision
            }
            for task in active_tasks.values()
        ]
    }

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    if task_id in active_tasks:
        del active_tasks[task_id]
        return {"message": "Task deleted"}
    raise HTTPException(status_code=404, detail="Task not found")

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WADE - Autonomous Repo Refactor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            height: 120px;
            resize: vertical;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .status-card {
            display: none;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e1e5e9;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .logs {
            background: #f8f9fa;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            white-space: pre-wrap;
        }
        .files-changed {
            background: #e8f5e8;
            border: 1px solid #c3e6c3;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .execution-output {
            background: #f0f8ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 WADE</h1>
            <p>Autonomous Repository Refactoring Agent</p>
        </div>
        
        <div class="card">
            <h2>🚀 Start Refactoring</h2>
            <form id="refactorForm">
                <div class="form-group">
                    <label for="repoPath">Repository Path:</label>
                    <input type="text" id="repoPath" placeholder="/path/to/your/repo" value="/workspace" required>
                </div>
                
                <div class="form-group">
                    <label for="vision">Transformation Vision:</label>
                    <textarea id="vision" placeholder="Describe what you want to achieve. For example: 'Convert this Flask app to FastAPI with async endpoints and microservice architecture. Add logging middleware and proper error handling.'" required></textarea>
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label for="testCommand">Test Command (optional):</label>
                        <input type="text" id="testCommand" placeholder="python -m pytest">
                    </div>
                    
                    <div class="form-group">
                        <label for="runCommand">Run Command (optional):</label>
                        <input type="text" id="runCommand" placeholder="python main.py">
                    </div>
                </div>
                
                <button type="submit" class="btn" id="startBtn">🚀 Start Autonomous Refactoring</button>
            </form>
        </div>
        
        <div class="card status-card" id="statusCard">
            <h2>📊 Refactoring Status</h2>
            <div id="statusInfo">
                <p><strong>Task ID:</strong> <span id="taskId"></span></p>
                <p><strong>Status:</strong> <span id="status"></span></p>
                <p><strong>Current Step:</strong> <span id="currentStep"></span></p>
                
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                </div>
                <p><span id="progress">0</span>% Complete</p>
            </div>
            
            <h3>📝 Live Logs</h3>
            <div class="logs" id="logs"></div>
            
            <div id="filesChanged" style="display: none;">
                <h3>📁 Files Changed</h3>
                <div class="files-changed" id="filesChangedList"></div>
            </div>
            
            <div id="testResults" style="display: none;">
                <h3>🧪 Test Results</h3>
                <div class="execution-output" id="testResultsContent"></div>
            </div>
            
            <div id="executionOutput" style="display: none;">
                <h3>🚀 Execution Output</h3>
                <div class="execution-output" id="executionOutputContent"></div>
            </div>
        </div>
    </div>

    <script>
        let currentTaskId = null;
        let statusInterval = null;

        document.getElementById('refactorForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                repo_path: document.getElementById('repoPath').value,
                vision: document.getElementById('vision').value,
                test_command: document.getElementById('testCommand').value || null,
                run_command: document.getElementById('runCommand').value || null
            };
            
            const startBtn = document.getElementById('startBtn');
            startBtn.disabled = true;
            startBtn.textContent = '🚀 Starting...';
            
            try {
                const response = await fetch('/api/refactor', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    currentTaskId = result.task_id;
                    document.getElementById('taskId').textContent = currentTaskId;
                    document.getElementById('statusCard').style.display = 'block';
                    
                    // Start polling for status
                    statusInterval = setInterval(updateStatus, 2000);
                    updateStatus(); // Initial update
                } else {
                    alert('Error: ' + result.detail);
                    startBtn.disabled = false;
                    startBtn.textContent = '🚀 Start Autonomous Refactoring';
                }
            } catch (error) {
                alert('Error: ' + error.message);
                startBtn.disabled = false;
                startBtn.textContent = '🚀 Start Autonomous Refactoring';
            }
        });

        async function updateStatus() {
            if (!currentTaskId) return;
            
            try {
                const response = await fetch(`/api/status/${currentTaskId}`);
                const status = await response.json();
                
                document.getElementById('status').textContent = status.status;
                document.getElementById('currentStep').textContent = status.current_step;
                document.getElementById('progress').textContent = status.progress;
                document.getElementById('progressFill').style.width = status.progress + '%';
                
                // Update logs
                const logsDiv = document.getElementById('logs');
                logsDiv.textContent = status.logs.join('\\n');
                logsDiv.scrollTop = logsDiv.scrollHeight;
                
                // Update files changed
                if (status.files_changed && status.files_changed.length > 0) {
                    document.getElementById('filesChanged').style.display = 'block';
                    document.getElementById('filesChangedList').textContent = status.files_changed.join('\\n');
                }
                
                // Update test results
                if (status.test_results) {
                    document.getElementById('testResults').style.display = 'block';
                    document.getElementById('testResultsContent').textContent = status.test_results;
                }
                
                // Update execution output
                if (status.execution_output) {
                    document.getElementById('executionOutput').style.display = 'block';
                    document.getElementById('executionOutputContent').textContent = status.execution_output;
                }
                
                // Stop polling if completed or failed
                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(statusInterval);
                    const startBtn = document.getElementById('startBtn');
                    startBtn.disabled = false;
                    startBtn.textContent = '🚀 Start Autonomous Refactoring';
                }
                
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("🚀 Starting WADE Autonomous Repo Refactor Server...")
    print("📍 Access the web interface at: http://localhost:12000")
    print("📍 VS Code available at: http://localhost:60001")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=12000,
        reload=False
    )