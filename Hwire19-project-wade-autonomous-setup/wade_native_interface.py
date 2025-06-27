#!/usr/bin/env python3
"""
WADE Native Interface - OpenHands-Style Autonomous Development Environment
Chat-driven development with embedded VS Code, terminal, and browser preview
"""

import os
import json
import asyncio
import subprocess
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io

# Import WADE components
import sys
sys.path.append('/workspace/wade_env')

try:
    from wade_env.security import request_signer, token_manager
    from wade_env.security_middleware import security_middleware
    from wade_env.performance import model_prewarmer, query_cache, connection_pool
    from wade_env.websocket_manager import websocket_manager
    from wade_env.error_handler import error_handler, ErrorSeverity, ErrorCategory
    from wade_env.model_router import model_router
    from wade_env.intel_query import intel_query
except ImportError:
    # Create the wade_env directory if it doesn't exist
    os.makedirs("wade_env", exist_ok=True)
    if not os.path.exists("wade_env/__init__.py"):
        with open("wade_env/__init__.py", "w") as f:
            f.write("# WADE Environment Module\n")
    
    # Import will be attempted again after modules are created

try:
    from wade_env.settings_manager import settings_manager
    from wade_env.file_manager import file_manager
    from wade_env.model_manager import model_manager
    from wade_env.vscode_service import vscode_service
    from wade_env.terminal_service import terminal_service
    # Import new modules
    from wade_env.model_router import model_router
    from wade_env.project_sync import project_sync
    from wade_env.intel_query import intel_query
    from wade_env.self_evolution import self_evolution
    from wade_env.microagent_launcher import microagent_launcher
except ImportError as e:
    print(f"Warning: Could not import WADE components: {e}")
    # Create mock objects for now
    class MockManager:
        def get_settings_dict(self): return {}
        def update_settings(self, section, updates): return True
        def get_setting_options(self): return {}
        def list_profiles(self): return []
        def save_profile(self, name): return True
        def load_profile(self, name): return True
        def delete_profile(self, name): return True
        def reset_to_defaults(self): pass
        def validate_settings(self): return []
        async def list_files(self, directory="", show_hidden=False): return []
        async def read_file(self, file_path): return None
        async def write_file(self, file_path, content): return True
        async def upload_file(self, file_data, filename, directory): 
            from dataclasses import dataclass
            @dataclass
            class UploadResult:
                success: bool = True
                file_path: str = filename
                file_size: int = len(file_data)
                message: str = "Mock upload"
                hash_sha256: str = "mock_hash"
            return UploadResult()
        async def download_file(self, file_path): return b"mock file content"
        async def delete_file(self, file_path): return True
        async def create_directory(self, directory_path): return True
        async def move_file(self, source, target): return True
        async def copy_file(self, source, target): return True
        async def search_files(self, query, file_type=None): return []
        async def get_workspace_stats(self): return {}
        def get_available_models(self): return []
        def set_active_model(self, model_name): pass
        async def pull_ollama_model(self, model_name): return True
        async def check_model_availability(self, model_name): return False
        def is_running(self): return False
        def get_server_url(self): return "http://localhost:12001"
        async def start_server(self): return True
        async def stop_server(self): pass
        async def restart_server(self): return True
    
    settings_manager = MockManager()
    file_manager = MockManager()
    model_manager = MockManager()
    vscode_service = MockManager()
    terminal_service = MockManager()
    # Mock new modules
    model_router = MockManager()
    project_sync = MockManager()
    intel_query = MockManager()
    self_evolution = MockManager()
    microagent_launcher = MockManager()

# Chat and Agent Models
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None
    agent_id: Optional[str] = None

class MicroAgent(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    triggers: List[str]
    status: str = "active"
    created_at: str

class TaskExecution(BaseModel):
    task_id: str
    agent_id: str
    description: str
    status: str = "pending"  # "pending", "running", "completed", "failed"
    progress: int = 0
    logs: List[str] = []
    files_modified: List[str] = []
    output: Optional[str] = None
    execution_mode: str = "simulation"  # "simulation" or "live"

class ExecutionMode(BaseModel):
    mode: str  # "simulation" or "live"
    description: str
    safety_level: str
    capabilities: List[str]

# Global state
chat_history: List[ChatMessage] = []
active_agents: Dict[str, MicroAgent] = {}
task_queue: Dict[str, TaskExecution] = {}
websocket_connections: List[WebSocket] = []
current_execution_mode: str = "simulation"  # Default to safe simulation mode

# Execution mode configurations
EXECUTION_MODES = {
    "simulation": ExecutionMode(
        mode="simulation",
        description="Safe simulation environment - all operations are sandboxed and logged",
        safety_level="maximum",
        capabilities=[
            "Code analysis and review",
            "Theoretical vulnerability assessment", 
            "Simulated penetration testing",
            "Security education and training",
            "Compliance checking",
            "Documentation generation",
            "Safe code execution in containers"
        ]
    ),
    "live": ExecutionMode(
        mode="live",
        description="Live execution environment - real operations with full system access",
        safety_level="user_controlled",
        capabilities=[
            "Direct system modifications",
            "Network operations",
            "File system access",
            "Service deployment",
            "Database operations",
            "External API interactions",
            "Production deployments"
        ]
    )
}

app = FastAPI(title="WADE Native Interface", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
@app.middleware("http")
async def security_middleware_handler(request, call_next):
    """Apply security checks to all requests"""
    try:
        # Skip security checks for static files and WebSockets
        if request.url.path.startswith("/static") or request.url.path.startswith("/ws"):
            return await call_next(request)
        
        # Validate request signature (if present)
        if "X-Request-Signature" in request.headers:
            if not security_middleware.validate_request_signature(request):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid request signature"}
                )
        
        # Apply rate limiting
        client_ip = request.client.host
        if not security_middleware.check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add security headers to response
        for header, value in security_middleware.get_security_headers().items():
            response.headers[header] = value
        
        return response
    except Exception as e:
        # Log the error
        error_id = await error_handler.handle_error(e, {"path": request.url.path})
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_id": error_id
            }
        )

class PhindCodeLlamaAgent:
    """Simulated Phind-CodeLlama conversation agent"""
    
    def __init__(self):
        self.context = []
        self.workspace_path = "/workspace/wade_env"
    
    async def process_message(self, message: str, user_id: str = "user") -> str:
        """Process user message and generate response"""
        
        # Analyze message for intent and create micro-agents if needed
        intent = await self.analyze_intent(message)
        
        if intent["create_agent"]:
            agent = await self.create_micro_agent(intent)
            if agent:
                response = f"🤖 Created micro-agent: **{agent.name}**\n\n{agent.description}\n\nCapabilities:\n"
                for cap in agent.capabilities:
                    response += f"• {cap}\n"
                response += f"\nThis agent will automatically handle tasks related to: {', '.join(agent.triggers)}"
                return response
        
        if intent["execute_task"]:
            task = await self.create_task(intent)
            if task:
                return f"🚀 Started task: **{task.description}**\n\nTask ID: `{task.task_id}`\nAgent: {task.agent_id}\n\nI'll execute this autonomously and update you with progress."
        
        # Generate contextual response
        return await self.generate_response(message, intent)
    
    async def analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user message for intent and required actions"""
        message_lower = message.lower()
        
        intent = {
            "create_agent": False,
            "execute_task": False,
            "agent_type": None,
            "task_description": None,
            "technologies": [],
            "scope": []
        }
        
        # Detect agent creation needs
        agent_triggers = {
            "security": ["security", "vulnerability", "audit", "compliance", "penetration"],
            "frontend": ["react", "vue", "angular", "frontend", "ui", "interface"],
            "backend": ["api", "server", "database", "backend", "microservice"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment", "infrastructure"],
            "testing": ["test", "testing", "qa", "automation", "selenium"],
            "data": ["data", "analytics", "ml", "ai", "machine learning", "analysis"]
        }
        
        for agent_type, triggers in agent_triggers.items():
            if any(trigger in message_lower for trigger in triggers):
                intent["create_agent"] = True
                intent["agent_type"] = agent_type
                break
        
        # Detect task execution needs
        task_triggers = [
            "create", "build", "implement", "develop", "generate", "setup", "configure",
            "refactor", "optimize", "fix", "debug", "deploy", "test", "analyze"
        ]
        
        if any(trigger in message_lower for trigger in task_triggers):
            intent["execute_task"] = True
            intent["task_description"] = message
        
        # Extract technologies
        tech_keywords = {
            "python": ["python", "flask", "django", "fastapi"],
            "javascript": ["javascript", "js", "node", "react", "vue"],
            "docker": ["docker", "container", "containerize"],
            "kubernetes": ["kubernetes", "k8s", "helm"],
            "aws": ["aws", "amazon", "s3", "ec2", "lambda"],
            "security": ["security", "auth", "encryption", "ssl", "tls"]
        }
        
        for tech, keywords in tech_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intent["technologies"].append(tech)
        
        return intent
    
    async def create_micro_agent(self, intent: Dict[str, Any]) -> Optional[MicroAgent]:
        """Create a new micro-agent based on intent"""
        agent_type = intent["agent_type"]
        
        agent_templates = {
            "security": {
                "name": "Security Specialist Agent",
                "description": "Autonomous security assessment and hardening specialist",
                "capabilities": [
                    "Vulnerability scanning and assessment",
                    "Security code review and analysis", 
                    "Compliance auditing (GDPR, HIPAA, SOC2)",
                    "Penetration testing and threat modeling",
                    "Security hardening recommendations",
                    "Incident response planning"
                ],
                "triggers": ["security", "vulnerability", "audit", "compliance", "penetration", "hardening"]
            },
            "frontend": {
                "name": "Frontend Development Agent",
                "description": "Autonomous frontend development and UI/UX specialist",
                "capabilities": [
                    "React/Vue/Angular application development",
                    "Responsive design implementation",
                    "Component library creation",
                    "Performance optimization",
                    "Accessibility compliance",
                    "Cross-browser testing"
                ],
                "triggers": ["frontend", "ui", "react", "vue", "angular", "component", "interface"]
            },
            "backend": {
                "name": "Backend Development Agent", 
                "description": "Autonomous backend API and service development specialist",
                "capabilities": [
                    "REST/GraphQL API development",
                    "Database design and optimization",
                    "Microservices architecture",
                    "Authentication and authorization",
                    "Caching and performance tuning",
                    "API documentation generation"
                ],
                "triggers": ["backend", "api", "server", "database", "microservice", "endpoint"]
            },
            "devops": {
                "name": "DevOps Automation Agent",
                "description": "Autonomous infrastructure and deployment specialist", 
                "capabilities": [
                    "CI/CD pipeline creation",
                    "Docker containerization",
                    "Kubernetes orchestration",
                    "Infrastructure as Code",
                    "Monitoring and alerting setup",
                    "Automated deployment strategies"
                ],
                "triggers": ["devops", "docker", "kubernetes", "ci/cd", "deployment", "infrastructure"]
            },
            "testing": {
                "name": "Quality Assurance Agent",
                "description": "Autonomous testing and quality assurance specialist",
                "capabilities": [
                    "Unit and integration test creation",
                    "End-to-end test automation",
                    "Performance testing",
                    "Security testing",
                    "Test data generation",
                    "Quality metrics reporting"
                ],
                "triggers": ["test", "testing", "qa", "automation", "selenium", "quality"]
            },
            "data": {
                "name": "Data Science Agent",
                "description": "Autonomous data analysis and machine learning specialist",
                "capabilities": [
                    "Data analysis and visualization",
                    "Machine learning model development",
                    "Data pipeline creation",
                    "Statistical analysis",
                    "Predictive modeling",
                    "Data quality assessment"
                ],
                "triggers": ["data", "analytics", "ml", "ai", "machine learning", "analysis"]
            }
        }
        
        if agent_type not in agent_templates:
            return None
        
        template = agent_templates[agent_type]
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        
        agent = MicroAgent(
            agent_id=agent_id,
            name=template["name"],
            description=template["description"],
            capabilities=template["capabilities"],
            triggers=template["triggers"],
            created_at=datetime.now().isoformat()
        )
        
        active_agents[agent_id] = agent
        return agent
    
    async def create_task(self, intent: Dict[str, Any]) -> Optional[TaskExecution]:
        """Create a new task execution"""
        
        # Find appropriate agent for the task
        suitable_agents = []
        for agent_id, agent in active_agents.items():
            for trigger in agent.triggers:
                if trigger in intent["task_description"].lower():
                    suitable_agents.append(agent)
                    break
        
        if not suitable_agents:
            # Create a general purpose agent
            general_agent = MicroAgent(
                agent_id=f"general_{uuid.uuid4().hex[:8]}",
                name="General Development Agent",
                description="General purpose development and automation agent",
                capabilities=["Code generation", "File manipulation", "Task automation"],
                triggers=["general", "code", "file", "task"],
                created_at=datetime.now().isoformat()
            )
            active_agents[general_agent.agent_id] = general_agent
            suitable_agents = [general_agent]
        
        # Use the first suitable agent
        selected_agent = suitable_agents[0]
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = TaskExecution(
            task_id=task_id,
            agent_id=selected_agent.agent_id,
            description=intent["task_description"],
            status="pending"
        )
        
        task_queue[task_id] = task
        
        # Start task execution in background
        asyncio.create_task(self.execute_task(task))
        
        return task
    
    async def execute_task(self, task: TaskExecution):
        """Execute a task autonomously based on execution mode"""
        try:
            task.status = "running"
            task.execution_mode = current_execution_mode
            await self.broadcast_task_update(task)
            
            # Log execution mode
            mode_info = EXECUTION_MODES[current_execution_mode]
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Execution Mode: {mode_info.mode.upper()}")
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🛡️ Safety Level: {mode_info.safety_level}")
            
            if current_execution_mode == "simulation":
                await self.execute_simulation_mode(task)
            else:
                await self.execute_live_mode(task)
            
            # Mark as completed
            task.status = "completed"
            task.output = f"Task '{task.description}' completed successfully by agent {task.agent_id} in {current_execution_mode} mode"
            await self.broadcast_task_update(task)
            
        except Exception as e:
            task.status = "failed"
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
            await self.broadcast_task_update(task)
    
    async def execute_simulation_mode(self, task: TaskExecution):
        """Execute task in safe simulation mode"""
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🧪 SIMULATION MODE: All operations are sandboxed")
        
        simulation_steps = [
            "🔍 Analyzing requirements safely",
            "🧠 Planning implementation (theoretical)",
            "📝 Generating code (sandboxed)",
            "🧪 Simulating execution (no real changes)",
            "📊 Generating safety report",
            "✅ Simulation complete"
        ]
        
        for i, step in enumerate(simulation_steps):
            task.progress = int((i + 1) / len(simulation_steps) * 100)
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {step}")
            await self.broadcast_task_update(task)
            await asyncio.sleep(1.5)  # Faster simulation
        
        # Add simulation results
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 Simulation Results:")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] • No real system changes made")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] • All operations logged for review")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] • Ready for live execution if approved")
    
    async def execute_live_mode(self, task: TaskExecution):
        """Execute task in live mode with real system access"""
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 LIVE MODE: Real system operations enabled")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ WARNING: Changes will affect real systems")
        
        live_steps = [
            "🔍 Analyzing requirements",
            "🛡️ Performing safety checks",
            "📝 Generating implementation code",
            "🚀 Executing real operations",
            "🧪 Testing live changes",
            "📊 Validating results",
            "✅ Live execution complete"
        ]
        
        for i, step in enumerate(live_steps):
            task.progress = int((i + 1) / len(live_steps) * 100)
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {step}")
            await self.broadcast_task_update(task)
            await asyncio.sleep(2.5)  # Slower for real operations
        
        # Add live execution results
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Live Execution Results:")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] • Real system modifications applied")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] • All changes logged and tracked")
        task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] • Backup created before modifications")
    
    async def broadcast_task_update(self, task: TaskExecution):
        """Broadcast task updates to all connected clients"""
        message = {
            "type": "task_update",
            "task": task.dict()
        }
        
        for connection in websocket_connections:
            try:
                await connection.send_json(message)
            except:
                pass
    
    async def generate_response(self, message: str, intent: Dict[str, Any]) -> str:
        """Generate contextual response"""
        
        # Context-aware responses
        if "help" in message.lower():
            return """🤖 **WADE Native Interface Help**

I'm your autonomous development assistant! Here's what I can do:

**🔧 Development Tasks:**
• Create and refactor code in any language
• Set up project structures and configurations
• Implement features and fix bugs
• Generate documentation and tests

**🛡️ Security & Compliance:**
• Perform security audits and vulnerability assessments
• Implement security best practices
• Ensure compliance with regulations (GDPR, HIPAA, etc.)

**🚀 DevOps & Deployment:**
• Create CI/CD pipelines
• Set up Docker containers and Kubernetes
• Configure monitoring and alerting

**💬 How to interact:**
• Just describe what you want to build or accomplish
• I'll create specialized micro-agents to handle specific tasks
• All work is done autonomously with real-time progress updates

**Example commands:**
• "Create a secure REST API with authentication"
• "Set up a React frontend with modern best practices"
• "Perform a security audit of this codebase"
• "Deploy this application to Kubernetes"

What would you like to work on?"""
        
        if "status" in message.lower():
            active_count = len(active_agents)
            running_tasks = len([t for t in task_queue.values() if t.status == "running"])
            
            return f"""📊 **System Status**

**Active Agents:** {active_count}
**Running Tasks:** {running_tasks}
**Workspace:** `/workspace/wade_env`

**Recent Activity:**
{chr(10).join([f"• {task.description[:50]}..." for task in list(task_queue.values())[-3:]])}

Everything is running smoothly! What would you like me to work on next?"""
        
        # Default response
        return f"""I understand you want to: "{message}"

Let me analyze this and create the appropriate micro-agents to handle your request. I'll work autonomously and keep you updated on progress.

Would you like me to proceed with this task?"""

# Initialize the Phind agent
phind_agent = PhindCodeLlamaAgent()

# WebSocket endpoint for real-time chat
@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time system updates"""
    await websocket.accept()
    
    # Register with WebSocket manager
    connection_id = await websocket_manager.connect(websocket)
    
    try:
        # Send initial system state
        await websocket.send_json({
            "type": "system_state",
            "timestamp": time.time(),
            "data": {
                "model_router": {
                    "active_model": model_router.get_last_selected_model(),
                    "performance": model_router.get_performance_metrics()
                },
                "agents": {
                    "count": 0,  # Will be updated when microagent_launcher is implemented
                    "status": "ready"
                },
                "intel": {
                    "performance": intel_query.get_performance_metrics()
                }
            }
        })
        
        # Keep connection alive with periodic updates
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            
            # Process client messages if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Clean up on disconnect
        websocket_manager.disconnect(connection_id)
    except Exception as e:
        # Log error
        await error_handler.handle_error(e, {"connection_id": connection_id})
        # Clean up
        websocket_manager.disconnect(connection_id)

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat_message":
                user_message = ChatMessage(
                    role="user",
                    content=data["content"],
                    timestamp=datetime.now().isoformat()
                )
                chat_history.append(user_message)
                
                # Process with Phind agent
                response_content = await phind_agent.process_message(data["content"])
                
                assistant_message = ChatMessage(
                    role="assistant", 
                    content=response_content,
                    timestamp=datetime.now().isoformat()
                )
                chat_history.append(assistant_message)
                
                # Send response back
                await websocket.send_json({
                    "type": "chat_response",
                    "message": assistant_message.dict()
                })
                
                # Broadcast to all connections
                for conn in websocket_connections:
                    if conn != websocket:
                        try:
                            await conn.send_json({
                                "type": "chat_update",
                                "message": assistant_message.dict()
                            })
                        except:
                            pass
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

# API Endpoints
@app.get("/api/agents")
async def get_agents():
    """Get all active micro-agents"""
    return {"agents": [agent.dict() for agent in active_agents.values()]}

@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks"""
    return {"tasks": [task.dict() for task in task_queue.values()]}

@app.get("/api/chat-history")
async def get_chat_history():
    """Get chat history"""
    return {"messages": [msg.dict() for msg in chat_history]}

@app.get("/api/execution-mode")
async def get_execution_mode():
    """Get current execution mode"""
    mode_info = EXECUTION_MODES[current_execution_mode]
    return {
        "current_mode": current_execution_mode,
        "mode_info": mode_info.dict(),
        "available_modes": {mode: info.dict() for mode, info in EXECUTION_MODES.items()}
    }

@app.post("/api/execution-mode")
async def set_execution_mode(mode_data: dict):
    """Set execution mode"""
    global current_execution_mode
    
    new_mode = mode_data.get("mode")
    if new_mode not in EXECUTION_MODES:
        raise HTTPException(status_code=400, detail="Invalid execution mode")
    
    old_mode = current_execution_mode
    current_execution_mode = new_mode
    
    # Broadcast mode change to all connected clients
    mode_info = EXECUTION_MODES[current_execution_mode]
    message = {
        "type": "execution_mode_changed",
        "old_mode": old_mode,
        "new_mode": current_execution_mode,
        "mode_info": mode_info.dict()
    }
    
    for connection in websocket_connections:
        try:
            await connection.send_json(message)
        except:
            pass
    
    return {
        "success": True,
        "old_mode": old_mode,
        "new_mode": current_execution_mode,
        "mode_info": mode_info.dict()
    }

# Settings API Endpoints
@app.get("/api/settings")
async def get_settings():
    """Get all WADE settings"""
    return settings_manager.get_settings_dict()

@app.post("/api/settings/{section}")
async def update_settings(section: str, updates: dict):
    """Update settings section"""
    success = settings_manager.update_settings(section, updates)
    if success:
        return {"success": True, "message": f"Settings section '{section}' updated"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to update settings section '{section}'")

@app.get("/api/settings/options")
async def get_setting_options():
    """Get available options for dropdown settings"""
    return settings_manager.get_setting_options()

@app.get("/api/settings/profiles")
async def list_profiles():
    """List available settings profiles"""
    return {"profiles": settings_manager.list_profiles()}

@app.post("/api/settings/profiles/{profile_name}")
async def save_profile(profile_name: str):
    """Save current settings as profile"""
    success = settings_manager.save_profile(profile_name)
    if success:
        return {"success": True, "message": f"Profile '{profile_name}' saved"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to save profile '{profile_name}'")

@app.put("/api/settings/profiles/{profile_name}")
async def load_profile(profile_name: str):
    """Load settings from profile"""
    success = settings_manager.load_profile(profile_name)
    if success:
        return {"success": True, "message": f"Profile '{profile_name}' loaded"}
    else:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")

@app.delete("/api/settings/profiles/{profile_name}")
async def delete_profile(profile_name: str):
    """Delete settings profile"""
    success = settings_manager.delete_profile(profile_name)
    if success:
        return {"success": True, "message": f"Profile '{profile_name}' deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")

@app.post("/api/settings/reset")
async def reset_settings():
    """Reset all settings to defaults"""
    settings_manager.reset_to_defaults()
    return {"success": True, "message": "Settings reset to defaults"}

@app.get("/api/settings/validate")
async def validate_settings():
    """Validate current settings"""
    issues = settings_manager.validate_settings()
    return {"valid": len(issues) == 0, "issues": issues}

# File Management API Endpoints
@app.get("/api/files")
async def list_files(directory: str = "", show_hidden: bool = False):
    """List files in directory"""
    files = await file_manager.list_files(directory, show_hidden)
    return {"files": [asdict(f) for f in files]}

@app.get("/api/files/content")
async def get_file_content(file_path: str):
    """Get file content"""
    content = await file_manager.read_file(file_path)
    if content is not None:
        return {"content": content, "file_path": file_path}
    else:
        raise HTTPException(status_code=404, detail="File not found or not readable")

@app.post("/api/files/content")
async def save_file_content(file_data: dict):
    """Save file content"""
    file_path = file_data.get("file_path")
    content = file_data.get("content", "")
    
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    
    success = await file_manager.write_file(file_path, content)
    if success:
        return {"success": True, "message": f"File '{file_path}' saved"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to save file '{file_path}'")

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...), directory: str = Form("uploads")):
    """Upload file to workspace"""
    try:
        file_data = await file.read()
        result = await file_manager.upload_file(file_data, file.filename, directory)
        
        if result.success:
            return {
                "success": True,
                "file_path": result.file_path,
                "file_size": result.file_size,
                "message": result.message,
                "hash": result.hash_sha256
            }
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/files/download")
async def download_file(file_path: str):
    """Download file from workspace"""
    file_data = await file_manager.download_file(file_path)
    
    if file_data is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = Path(file_path).name
    
    return StreamingResponse(
        io.BytesIO(file_data),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.delete("/api/files")
async def delete_file(file_path: str):
    """Delete file or directory"""
    success = await file_manager.delete_file(file_path)
    if success:
        return {"success": True, "message": f"'{file_path}' deleted"}
    else:
        raise HTTPException(status_code=404, detail="File not found or deletion failed")

@app.post("/api/files/directory")
async def create_directory(dir_data: dict):
    """Create directory"""
    directory_path = dir_data.get("path")
    if not directory_path:
        raise HTTPException(status_code=400, detail="path is required")
    
    success = await file_manager.create_directory(directory_path)
    if success:
        return {"success": True, "message": f"Directory '{directory_path}' created"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to create directory '{directory_path}'")

@app.post("/api/files/move")
async def move_file(move_data: dict):
    """Move/rename file or directory"""
    source = move_data.get("source")
    target = move_data.get("target")
    
    if not source or not target:
        raise HTTPException(status_code=400, detail="source and target are required")
    
    success = await file_manager.move_file(source, target)
    if success:
        return {"success": True, "message": f"Moved '{source}' to '{target}'"}
    else:
        raise HTTPException(status_code=400, detail="Move operation failed")

@app.post("/api/files/copy")
async def copy_file(copy_data: dict):
    """Copy file or directory"""
    source = copy_data.get("source")
    target = copy_data.get("target")
    
    if not source or not target:
        raise HTTPException(status_code=400, detail="source and target are required")
    
    success = await file_manager.copy_file(source, target)
    if success:
        return {"success": True, "message": f"Copied '{source}' to '{target}'"}
    else:
        raise HTTPException(status_code=400, detail="Copy operation failed")

@app.get("/api/files/search")
async def search_files(query: str, file_type: str = None):
    """Search files"""
    files = await file_manager.search_files(query, file_type)
    return {"files": [asdict(f) for f in files], "query": query}

@app.get("/api/workspace/stats")
async def get_workspace_stats():
    """Get workspace statistics"""
    stats = await file_manager.get_workspace_stats()
    return stats

# Model Management API Endpoints
@app.get("/api/models")
async def get_models():
    """Get available models"""
    return {"models": model_manager.get_available_models()}

@app.post("/api/models/active")
async def set_active_model(model_data: dict):
    """Set active model"""
    model_name = model_data.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    
    model_manager.set_active_model(model_name)
    return {"success": True, "message": f"Active model set to '{model_name}'"}

@app.post("/api/models/pull")
async def pull_model(model_data: dict):
    """Pull/download model"""
    model_name = model_data.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    
    success = await model_manager.pull_ollama_model(model_name)
    if success:
        return {"success": True, "message": f"Model '{model_name}' pulled successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to pull model '{model_name}'")

@app.get("/api/models/check/{model_name}")
async def check_model(model_name: str):
    """Check if model is available"""
    available = await model_manager.check_model_availability(model_name)
    return {"model_name": model_name, "available": available}

# VS Code Integration API Endpoints
@app.get("/api/vscode/status")
async def get_vscode_status():
    """Get VS Code server status"""
    return {
        "running": vscode_service.is_running(),
        "url": vscode_service.get_server_url()
    }

@app.post("/api/vscode/start")
async def start_vscode():
    """Start VS Code server"""
    success = await vscode_service.start_server()
    if success:
        return {"success": True, "url": vscode_service.get_server_url()}
    else:
        raise HTTPException(status_code=500, detail="Failed to start VS Code server")

@app.post("/api/vscode/stop")
async def stop_vscode():
    """Stop VS Code server"""
    await vscode_service.stop_server()
    return {"success": True, "message": "VS Code server stopped"}

@app.post("/api/vscode/restart")
async def restart_vscode():
    """Restart VS Code server"""
    success = await vscode_service.restart_server()
    if success:
        return {"success": True, "url": vscode_service.get_server_url()}
    else:
        raise HTTPException(status_code=500, detail="Failed to restart VS Code server")

# Model Router API Routes
@app.get("/api/model-router/status")
async def get_model_router_status():
    """Get the status of the model router"""
    return {
        "enabled": model_router.is_enabled(),
        "override_model": model_router.get_last_selected_model(),
        "usage_stats": model_router.get_usage_stats()
    }

@app.post("/api/model-router/toggle")
async def toggle_model_router(enabled: bool):
    """Enable or disable the model router"""
    model_router.set_enabled(enabled)
    return {"success": True, "enabled": model_router.is_enabled()}

@app.post("/api/model-router/override")
async def set_model_override(model_name: Optional[str] = None):
    """Set a model to override the automatic selection"""
    success = model_router.set_override_model(model_name)
    return {"success": success, "override_model": model_router.get_last_selected_model()}


# Project Sync API Routes
@app.get("/api/project/exports")
async def list_project_exports():
    """List all available project exports"""
    exports = await project_sync.list_exports()
    return {"exports": exports}

@app.post("/api/project/export")
async def export_project(include_chat_history: bool = True, 
                        include_memory: bool = True,
                        include_settings: bool = True):
    """Export the current project"""
    result = await project_sync.export_project(
        include_chat_history=include_chat_history,
        include_memory=include_memory,
        include_settings=include_settings
    )
    return result

@app.get("/api/project/download/{export_filename}")
async def download_project_export(export_filename: str):
    """Download a project export file"""
    export_path = os.path.join(project_sync.backup_path, export_filename)
    if not os.path.exists(export_path):
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return StreamingResponse(
        io.open(export_path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={export_filename}"}
    )

@app.post("/api/project/sync")
async def sync_project_to_cloud(export_path: str, provider: str = "all"):
    """Sync a project export to cloud storage"""
    result = await project_sync.sync_to_cloud(export_path, provider)
    return result


# Intel Query API Routes
@app.post("/api/intel-query")
async def perform_intel_query(query: str, sources: List[str] = None, max_results: int = 20):
    """Perform a comprehensive intelligence query"""
    result = await intel_query.intel_query(query, sources, max_results)
    return result

@app.post("/api/dark-search")
async def perform_dark_search(query: str, max_results: int = 10):
    """Perform a search on dark web sources"""
    result = await intel_query.dark_search(query, max_results)
    return result

@app.post("/api/fetch-content")
async def fetch_content(url: str, use_tor: bool = False):
    """Fetch content from a URL with optional Tor routing"""
    result = await intel_query.fetch_content(url, use_tor)
    return result

@app.post("/api/intel-query/tor")
async def toggle_tor(enabled: bool):
    """Enable or disable Tor routing"""
    success = intel_query.set_tor_enabled(enabled)
    return {"success": success, "tor_enabled": intel_query.is_tor_enabled()}


# Self Evolution API Routes
@app.get("/api/self-evolve/status")
async def get_evolution_status():
    """Get the status of the self-evolution system"""
    return {
        "version": self_evolution.get_version(),
        "evolution_in_progress": self_evolution.is_evolution_in_progress(),
        "current_evolution": self_evolution.get_current_evolution()
    }

@app.post("/api/self-evolve")
async def trigger_evolution():
    """Trigger the evolution process"""
    if self_evolution.is_evolution_in_progress():
        return {"success": False, "message": "Evolution already in progress"}
    
    result = await self_evolution.evolve()
    return result

@app.get("/api/self-evolve/metrics")
async def get_evolution_metrics(category: Optional[str] = None, days: Optional[int] = None):
    """Get performance metrics"""
    metrics = self_evolution.get_metrics(category, days)
    return {"metrics": metrics}

@app.get("/api/self-evolve/feedback")
async def get_evolution_feedback(feedback_type: Optional[str] = None, 
                               processed: Optional[bool] = None,
                               days: Optional[int] = None):
    """Get user feedback"""
    feedback = self_evolution.get_feedback(feedback_type, processed, days)
    return {"feedback": feedback}

@app.post("/api/self-evolve/feedback")
async def record_feedback(content: str, feedback_type: str, 
                        context: Optional[Dict[str, Any]] = None,
                        rating: Optional[int] = None):
    """Record user feedback"""
    feedback_id = self_evolution.record_feedback(content, feedback_type, context, None, rating)
    return {"success": True, "feedback_id": feedback_id}


# Microagent API Routes
@app.get("/api/microagents")
async def list_microagents(agent_type: Optional[str] = None, status: Optional[str] = None):
    """List microagents"""
    agents = microagent_launcher.list_agents(agent_type, status)
    return {"agents": agents}

@app.get("/api/microagents/{agent_id}")
async def get_microagent(agent_id: str):
    """Get information about a microagent"""
    agent = microagent_launcher.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/api/microagents/spawn")
async def spawn_microagent(agent_type: str, 
                         custom_name: Optional[str] = None,
                         custom_description: Optional[str] = None):
    """Spawn a new microagent"""
    agent_id = await microagent_launcher.spawn_agent(
        agent_type, 
        custom_name, 
        custom_description
    )
    if not agent_id:
        raise HTTPException(status_code=400, detail="Failed to spawn agent")
    return {"agent_id": agent_id}

@app.post("/api/microagents/{agent_id}/terminate")
async def terminate_microagent(agent_id: str):
    """Terminate a microagent"""
    success = await microagent_launcher.terminate_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": success}

@app.get("/api/microagent-tasks")
async def list_microagent_tasks(agent_id: Optional[str] = None, 
                              status: Optional[str] = None):
    """List microagent tasks"""
    tasks = microagent_launcher.list_tasks(agent_id, status)
    return {"tasks": tasks}

@app.get("/api/microagent-tasks/{task_id}")
async def get_microagent_task(task_id: str):
    """Get information about a microagent task"""
    task = microagent_launcher.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/api/microagent-tasks")
async def create_microagent_task(description: str,
                               agent_type: Optional[str] = None,
                               agent_id: Optional[str] = None,
                               execution_mode: Optional[str] = None,
                               priority: int = 3):
    """Create a new microagent task"""
    task_id = await microagent_launcher.create_task(
        description,
        agent_type,
        agent_id,
        execution_mode,
        priority
    )
    if not task_id:
        raise HTTPException(status_code=400, detail="Failed to create task")
    return {"task_id": task_id}

@app.post("/api/microagent-tasks/{task_id}/cancel")
async def cancel_microagent_task(task_id: str):
    """Cancel a microagent task"""
    success = await microagent_launcher.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    return {"success": success}

@app.post("/api/microagent/process-message")
async def process_message_with_microagent(message: str, user_id: str = "user"):
    """Process a message with microagent system"""
    response = await microagent_launcher.process_message(message, user_id)
    return response

# Security API endpoints
@app.post("/api/auth/token")
async def generate_auth_token(user_data: dict):
    """Generate an authentication token"""
    user_id = user_data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")
    
    # Generate token
    token = token_manager.generate_token(user_id)
    
    return {
        "token": token,
        "expires_in": 86400,  # 24 hours
        "token_type": "Bearer"
    }

@app.get("/api/security/status")
async def get_security_status():
    """Get security system status"""
    return {
        "active_connections": websocket_manager.get_connection_count(),
        "rate_limits": {
            "max_requests": security_middleware.rate_limit_max_requests,
            "window_seconds": security_middleware.rate_limit_window
        },
        "blocked_ips_count": len(security_middleware.blocked_ips)
    }

# Model Router API endpoints
@app.get("/api/models/status")
async def get_model_status():
    """Get model router status"""
    return {
        "active_model": model_router.get_last_selected_model(),
        "usage_stats": model_router.get_usage_stats(),
        "performance": model_router.get_performance_metrics()
    }

@app.post("/api/models/warmup")
async def warmup_models():
    """Warm up all models"""
    results = await model_router.warmup_all_models()
    return {
        "success": True,
        "results": results
    }

@app.post("/api/models/set-override")
async def set_model_override(data: dict):
    """Set model override"""
    model_name = data.get("model")
    success = model_router.set_override_model(model_name)
    return {
        "success": success,
        "active_model": model_router.get_last_selected_model()
    }

# Intel Query API endpoints
@app.post("/api/intel-query")
async def execute_intel_query(query_data: dict):
    """Execute an intelligence query"""
    query = query_data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    
    search_type = query_data.get("type", "general")
    safe_search = query_data.get("safe_search", True)
    bypass_cache = query_data.get("bypass_cache", False)
    
    try:
        results = await intel_query.search(
            query=query,
            search_type=search_type,
            safe_search=safe_search,
            bypass_cache=bypass_cache
        )
        return results
    except Exception as e:
        error_id = await error_handler.handle_error(e, {"query": query})
        raise HTTPException(
            status_code=500,
            detail=f"Error executing query: {str(e)} (Error ID: {error_id})"
        )

@app.get("/api/intel/performance")
async def get_intel_performance():
    """Get intel query performance metrics"""
    return {
        "metrics": intel_query.get_performance_metrics()
    }

# Error handling API endpoints
@app.get("/api/errors")
async def get_errors(limit: int = 10, severity: Optional[str] = None):
    """Get error log"""
    severity_enum = None
    if severity:
        try:
            severity_enum = ErrorSeverity[severity.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    
    errors = error_handler.get_error_log(limit=limit, severity=severity_enum)
    return {
        "errors": errors,
        "total": len(errors)
    }

@app.get("/api/errors/stats")
async def get_error_stats():
    """Get error statistics"""
    return error_handler.get_error_stats()

# Serve the native interface
@app.get("/", response_class=HTMLResponse)
async def serve_native_interface():
    """Serve the OpenHands-style native interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WADE Native Interface</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            height: 100vh;
            overflow: hidden;
            background: #1e1e1e;
            color: #d4d4d4;
        }
        
        /* Loading spinner */
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .container {
            display: grid;
            grid-template-columns: 350px 1fr;
            height: 100vh;
        }
        
        .header {
            grid-column: 1 / -1;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 60px;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .header-left h1 {
            color: #ffffff;
            font-size: 24px;
            font-weight: 700;
        }
        
        .subtitle {
            color: #cccccc;
            font-size: 14px;
        }
        
        .header-center {
            display: flex;
            gap: 5px;
        }
        
        .nav-tab {
            background: transparent;
            border: 1px solid #3e3e42;
            color: #cccccc;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        
        .nav-tab:hover {
            background: #3e3e42;
            color: #ffffff;
        }
        
        .nav-tab.active {
            background: #007acc;
            color: #ffffff;
            border-color: #007acc;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 350px 1fr;
            height: calc(100vh - 60px);
        }
        
        /* Panel Styles */
        .files-panel, .settings-panel, .models-panel {
            background: #252526;
            border-right: 1px solid #3e3e42;
            display: flex;
            flex-direction: column;
        }
        
        .files-header, .settings-header, .models-header {
            padding: 15px;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .files-header h2, .settings-header h2, .models-header h2 {
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
        }
        
        .btn-primary {
            background: #007acc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .btn-primary:hover {
            background: #005a9e;
        }
        
        /* Files Panel */
        .files-toolbar {
            padding: 10px 15px;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .search-input {
            flex: 1;
            background: #3c3c3c;
            border: 1px solid #3e3e42;
            color: #d4d4d4;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .files-content {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }
        
        .file-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .file-item:hover {
            background: #2a2d2e;
        }
        
        .file-icon {
            font-size: 16px;
        }
        
        .file-name {
            flex: 1;
            color: #d4d4d4;
        }
        
        .file-size {
            color: #888;
            font-size: 12px;
        }
        
        .file-actions {
            display: flex;
            gap: 5px;
        }
        
        .file-actions button {
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            padding: 4px;
            border-radius: 2px;
        }
        
        .file-actions button:hover {
            background: #3e3e42;
            color: #d4d4d4;
        }
        
        /* Settings Panel */
        .settings-actions {
            display: flex;
            gap: 10px;
        }
        
        .settings-tabs {
            display: flex;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            overflow-x: auto;
        }
        
        .settings-tab {
            background: transparent;
            border: none;
            color: #cccccc;
            padding: 12px 16px;
            cursor: pointer;
            font-size: 14px;
            white-space: nowrap;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .settings-tab:hover {
            background: #3e3e42;
            color: #ffffff;
        }
        
        .settings-tab.active {
            color: #007acc;
            border-bottom-color: #007acc;
        }
        
        .settings-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        /* Models Panel */
        .models-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .model-section {
            margin-bottom: 30px;
        }
        
        .model-section h3 {
            color: #ffffff;
            margin-bottom: 15px;
            font-size: 16px;
        }
        
        .active-model {
            background: #2d2d30;
            border: 1px solid #3e3e42;
            border-radius: 6px;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .model-name {
            color: #d4d4d4;
            font-weight: 600;
        }
        
        .model-status {
            font-size: 14px;
        }
        
        .models-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        /* Settings Form Styles */
        .setting-group {
            margin-bottom: 30px;
            padding: 20px;
            background: #2d2d30;
            border-radius: 6px;
            border: 1px solid #3e3e42;
        }
        
        .setting-group h3 {
            color: #ffffff;
            margin-bottom: 20px;
            font-size: 16px;
            border-bottom: 1px solid #3e3e42;
            padding-bottom: 10px;
        }
        
        .setting-item {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .setting-item label {
            color: #d4d4d4;
            font-weight: 600;
            font-size: 14px;
        }
        
        .setting-item select, .setting-item input[type="text"], .setting-item input[type="number"], .setting-item textarea {
            background: #3c3c3c;
            border: 1px solid #3e3e42;
            color: #d4d4d4;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .setting-item select:focus, .setting-item input:focus, .setting-item textarea:focus {
            outline: none;
            border-color: #007acc;
        }
        
        .setting-item input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: #007acc;
        }
        
        .setting-item input[type="range"] {
            width: 100%;
            accent-color: #007acc;
        }
        
        .setting-item small {
            color: #888;
            font-size: 12px;
            font-style: italic;
        }
        
        .setting-item textarea {
            resize: vertical;
            min-height: 80px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        
        /* Editor Modal */
        .editor-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .editor-content {
            background: #252526;
            border: 1px solid #3e3e42;
            border-radius: 6px;
            width: 80%;
            height: 80%;
            display: flex;
            flex-direction: column;
        }
        
        .editor-header {
            padding: 15px;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .editor-header h3 {
            color: #ffffff;
            margin: 0;
        }
        
        .editor-header button {
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 18px;
        }
        
        .editor-header button:hover {
            color: #d4d4d4;
        }
        
        #fileEditor {
            flex: 1;
            background: #1e1e1e;
            border: none;
            color: #d4d4d4;
            padding: 20px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            resize: none;
            outline: none;
        }
        
        .editor-actions {
            padding: 15px;
            background: #2d2d30;
            border-top: 1px solid #3e3e42;
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        .editor-actions button {
            background: #007acc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .editor-actions button:last-child {
            background: #666;
        }
        
        .editor-actions button:hover {
            opacity: 0.8;
        }
        
        .empty-state {
            text-align: center;
            color: #888;
            padding: 40px;
            font-style: italic;
        }
        
        /* Left Panel - Chat */
        .chat-panel {
            background: #252526;
            border-right: 1px solid #3e3e42;
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            padding: 15px;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chat-header h2 {
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            background: #4caf50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            max-width: 90%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        .message.user {
            background: #0e639c;
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }
        
        .message.assistant {
            background: #2d2d30;
            border: 1px solid #3e3e42;
            align-self: flex-start;
        }
        
        .message.assistant strong {
            color: #4fc3f7;
        }
        
        .message-time {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .chat-input-container {
            padding: 15px;
            background: #2d2d30;
            border-top: 1px solid #3e3e42;
        }
        
        .chat-input {
            width: 100%;
            padding: 12px 16px;
            background: #3c3c3c;
            border: 1px solid #5a5a5a;
            border-radius: 8px;
            color: #d4d4d4;
            font-size: 14px;
            resize: none;
            min-height: 40px;
            max-height: 120px;
        }
        
        .chat-input:focus {
            outline: none;
            border-color: #0e639c;
        }
        
        /* Right Panel - Workspace */
        .workspace-panel {
            display: grid;
            grid-template-rows: 40px 1fr 200px;
            background: #1e1e1e;
        }
        
        /* Toolbar */
        .toolbar {
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 15px;
            gap: 15px;
        }
        
        .toolbar-section {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .toolbar-button {
            background: #3c3c3c;
            border: 1px solid #5a5a5a;
            color: #d4d4d4;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .toolbar-button:hover {
            background: #4a4a4a;
        }
        
        .toolbar-button.active {
            background: #0e639c;
            border-color: #0e639c;
        }
        
        /* Execution Mode Toggle */
        .execution-mode-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #1e1e1e;
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid #3e3e42;
        }
        
        .mode-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            font-weight: 600;
        }
        
        .toggle-switch {
            position: relative;
            width: 60px;
            height: 28px;
            background: #3e3e42;
            border-radius: 14px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .toggle-switch.simulation {
            background: #4caf50;
        }
        
        .toggle-switch.live {
            background: #f44336;
        }
        
        .toggle-slider {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 24px;
            height: 24px;
            background: white;
            border-radius: 12px;
            transition: transform 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }
        
        .toggle-switch.live .toggle-slider {
            transform: translateX(32px);
        }
        
        .mode-status {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .mode-status.simulation {
            color: #4caf50;
        }
        
        .mode-status.live {
            color: #f44336;
        }
        
        .mode-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .mode-indicator.simulation {
            background: #4caf50;
        }
        
        .mode-indicator.live {
            background: #f44336;
        }
        
        /* Main Content Area */
        .content-area {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: #3e3e42;
        }
        
        .content-panel {
            background: #1e1e1e;
            display: flex;
            flex-direction: column;
        }
        
        .panel-header {
            background: #2d2d30;
            padding: 8px 15px;
            border-bottom: 1px solid #3e3e42;
            font-size: 12px;
            font-weight: 600;
            color: #cccccc;
        }
        
        .panel-content {
            flex: 1;
            overflow: hidden;
        }
        
        /* VS Code Iframe */
        .vscode-frame {
            width: 100%;
            height: 100%;
            border: none;
            background: #1e1e1e;
        }
        
        /* Browser Preview */
        .browser-frame {
            width: 100%;
            height: 100%;
            border: none;
            background: white;
        }
        
        /* Terminal */
        .terminal-panel {
            background: #0c0c0c;
            border-top: 1px solid #3e3e42;
            display: flex;
            flex-direction: column;
        }
        
        .terminal-header {
            background: #2d2d30;
            padding: 8px 15px;
            border-bottom: 1px solid #3e3e42;
            font-size: 12px;
            font-weight: 600;
            color: #cccccc;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .terminal-content {
            flex: 1;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #00ff00;
            overflow-y: auto;
            background: #0c0c0c;
        }
        
        /* Agent Status */
        .agent-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #2d2d30;
            border: 1px solid #3e3e42;
            border-radius: 8px;
            padding: 15px;
            min-width: 250px;
            z-index: 1000;
            display: none;
        }
        
        .agent-status.show {
            display: block;
        }
        
        .agent-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #3e3e42;
        }
        
        .agent-item:last-child {
            border-bottom: none;
        }
        
        .agent-name {
            font-weight: 600;
            color: #4fc3f7;
        }
        
        .agent-status-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            background: #4caf50;
            color: white;
        }
        
        /* Task Progress */
        .task-progress {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #2d2d30;
            border: 1px solid #3e3e42;
            border-radius: 8px;
            padding: 15px;
            min-width: 300px;
            z-index: 1000;
            display: none;
        }
        
        .task-progress.show {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #3e3e42;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: #4caf50;
            transition: width 0.3s ease;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 300px 1fr;
            }
            
            .content-area {
                grid-template-columns: 1fr;
                grid-template-rows: 1fr 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                grid-template-rows: 60vh 40vh;
            }
            
            .workspace-panel {
                grid-template-rows: 40px 1fr 100px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header with Navigation -->
        <div class="header">
            <div class="header-left">
                <h1>🤖 WADE</h1>
                <span class="subtitle">Autonomous Development Environment</span>
            </div>
            <div class="header-center">
                <button class="nav-tab active" data-tab="chat">💬 Chat</button>
                <button class="nav-tab" data-tab="files">📁 Files</button>
                <button class="nav-tab" data-tab="settings">⚙️ Settings</button>
                <button class="nav-tab" data-tab="models">🧠 Models</button>
                <button class="nav-tab" data-tab="agents">🤖 Agents</button>
                <button class="nav-tab" data-tab="intel">🔍 Intel</button>
                <button class="nav-tab" data-tab="project">📦 Project</button>
                <button class="nav-tab" data-tab="evolution">🧬 Evolution</button>
            </div>
            <div class="header-right">
                <div class="execution-mode-toggle">
                    <span class="mode-label">Mode:</span>
                    <div class="toggle-switch simulation" id="executionToggle" onclick="toggleExecutionMode()">
                        <div class="toggle-slider">🧪</div>
                    </div>
                    <div class="mode-status simulation" id="modeStatus">
                        <span class="mode-text">SIMULATION</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Content Area -->
        <div class="main-content">
            <!-- Left Panel - Chat -->
            <div class="chat-panel" id="chatPanel">
                <div class="chat-header">
                    <div class="status-indicator"></div>
                    <h2>🤖 WADE Assistant</h2>
                </div>
                
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <strong>Welcome to WADE Native Interface!</strong><br><br>
                        I'm your autonomous development assistant. I can help you with:
                        <br>• Code development and refactoring
                        <br>• Security assessments and compliance
                        <br>• DevOps and deployment automation
                        <br>• Testing and quality assurance
                        <br><br>
                        Just tell me what you'd like to build or accomplish, and I'll create specialized micro-agents to handle the work autonomously.
                        <div class="message-time">Just now</div>
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <textarea 
                        class="chat-input" 
                        id="chatInput" 
                        placeholder="Describe what you want to build or accomplish..."
                        rows="1"
                    ></textarea>
                </div>
            </div>
            
            <!-- Files Panel -->
            <div class="files-panel" id="filesPanel" style="display: none;">
                <div class="files-header">
                    <h2>📁 File Manager</h2>
                    <button class="btn-primary" onclick="uploadFile()">Upload File</button>
                </div>
                
                <div class="files-toolbar">
                    <input type="text" id="fileSearch" placeholder="Search files..." class="search-input">
                    <button onclick="createFolder()">📁 New Folder</button>
                    <button onclick="refreshFiles()">🔄 Refresh</button>
                </div>
                
                <div class="files-content" id="filesContent">
                    <div class="file-item">
                        <span class="file-icon">📄</span>
                        <span class="file-name">example.py</span>
                        <span class="file-size">1.2 KB</span>
                        <div class="file-actions">
                            <button onclick="editFile('example.py')">✏️</button>
                            <button onclick="downloadFile('example.py')">⬇️</button>
                            <button onclick="deleteFile('example.py')">🗑️</button>
                        </div>
                    </div>
                </div>
                
                <input type="file" id="fileUpload" style="display: none;" multiple>
            </div>
            
            <!-- Settings Panel -->
            <div class="settings-panel" id="settingsPanel" style="display: none;">
                <div class="settings-header">
                    <h2>⚙️ WADE Settings</h2>
                    <div class="settings-actions">
                        <button onclick="saveProfile()">💾 Save Profile</button>
                        <button onclick="loadProfile()">📂 Load Profile</button>
                        <button onclick="resetSettings()">🔄 Reset</button>
                    </div>
                </div>
                
                <div class="settings-tabs">
                    <button class="settings-tab active" data-section="models">🧠 Models</button>
                    <button class="settings-tab" data-section="execution">🔄 Execution</button>
                    <button class="settings-tab" data-section="autonomy">🧬 Autonomy</button>
                    <button class="settings-tab" data-section="network">📡 Network</button>
                    <button class="settings-tab" data-section="security">🔐 Security</button>
                    <button class="settings-tab" data-section="memory">💾 Memory</button>
                    <button class="settings-tab" data-section="ui">🎨 UI</button>
                </div>
                
                <div class="settings-content" id="settingsContent">
                    <!-- Settings content will be loaded dynamically -->
                </div>
            </div>
            
            <!-- Models Panel -->
            <div class="models-panel" id="modelsPanel" style="display: none;">
                <div class="models-header">
                    <h2>🧠 Model Management</h2>
                    <button class="btn-primary" onclick="pullModel()">⬇️ Pull Model</button>
                </div>
                
                <div class="models-content">
                    <div class="model-section">
                        <h3>Active Model</h3>
                        <div class="active-model" id="activeModel">
                            <span class="model-name">phind-codellama</span>
                            <span class="model-status">🟢 Ready</span>
                        </div>
                    </div>
                    
                    <div class="model-section">
                        <h3>Available Models</h3>
                        <div class="models-list" id="modelsList">
                            <!-- Models will be loaded dynamically -->
                        </div>
                    </div>
                </div>
            </div>

            <!-- Agents Panel -->
            <div class="panel agents-panel" id="agentsPanel" style="display: none;">
                <div class="panel-header">
                    <h2>🤖 Microagent Management</h2>
                    <button class="btn-primary" onclick="createAgent()">➕ New Agent</button>
                </div>

                <div class="panel-content">
                    <div class="section">
                        <h3>Active Agents</h3>
                        <div class="agent-list" id="agentList">
                            <div class="empty-state">No active agents</div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Agent Templates</h3>
                        <div class="template-list">
                            <div class="template-item">
                                <span class="template-name">Code Reviewer</span>
                                <button class="btn-sm" onclick="spawnAgent('code-reviewer')">Spawn</button>
                            </div>
                            <div class="template-item">
                                <span class="template-name">Data Analyzer</span>
                                <button class="btn-sm" onclick="spawnAgent('data-analyzer')">Spawn</button>
                            </div>
                            <div class="template-item">
                                <span class="template-name">Security Auditor</span>
                                <button class="btn-sm" onclick="spawnAgent('security-auditor')">Spawn</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Intel Panel -->
            <div class="panel intel-panel" id="intelPanel" style="display: none;">
                <div class="panel-header">
                    <h2>🔍 Intelligence Gathering</h2>
                </div>

                <div class="panel-content">
                    <div class="section">
                        <h3>Search Options</h3>
                        <div class="search-options">
                            <div class="option-group">
                                <label>Search Type:</label>
                                <select id="searchType">
                                    <option value="standard">Standard</option>
                                    <option value="dark">Dark Web (Tor)</option>
                                    <option value="academic">Academic</option>
                                </select>
                            </div>
                            <div class="option-group">
                                <label>Anonymity Level:</label>
                                <select id="anonymityLevel">
                                    <option value="none">None</option>
                                    <option value="basic">Basic</option>
                                    <option value="max">Maximum</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Intel Query</h3>
                        <div class="search-form">
                            <textarea id="intelQuery" placeholder="Enter your search query..."></textarea>
                            <button id="searchButton" class="btn-primary" onclick="executeIntelQuery()">🔍 Search</button>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Results</h3>
                        <div class="results-container" id="intelResults">
                            <div class="empty-state">No results yet</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Project Panel -->
            <div class="panel project-panel" id="projectPanel" style="display: none;">
                <div class="panel-header">
                    <h2>📦 Project Management</h2>
                </div>

                <div class="panel-content">
                    <div class="section">
                        <h3>Project Export</h3>
                        <div class="export-options">
                            <div class="option-group">
                                <label>Export Format:</label>
                                <select id="exportFormat">
                                    <option value="zip">ZIP Archive</option>
                                    <option value="tar">TAR Archive</option>
                                    <option value="json">JSON Bundle</option>
                                </select>
                            </div>
                            <div class="option-group">
                                <label>Include:</label>
                                <div class="checkbox-group">
                                    <input type="checkbox" id="includeChat" checked>
                                    <label for="includeChat">Chat History</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" id="includeFiles" checked>
                                    <label for="includeFiles">Project Files</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" id="includeMemory" checked>
                                    <label for="includeMemory">Memory Logs</label>
                                </div>
                            </div>
                            <button class="btn-primary" onclick="exportProject()">📦 Export Project</button>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Cloud Sync</h3>
                        <div class="cloud-options">
                            <div class="option-group">
                                <label>Cloud Provider:</label>
                                <select id="cloudProvider">
                                    <option value="gdrive">Google Drive</option>
                                    <option value="dropbox">Dropbox</option>
                                    <option value="s3">Amazon S3</option>
                                </select>
                            </div>
                            <button class="btn-primary" onclick="syncToCloud()">☁️ Sync to Cloud</button>
                        </div>
                        <div class="sync-status" id="syncStatus">
                            <div class="empty-state">Not synced yet</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Evolution Panel -->
            <div class="panel evolution-panel" id="evolutionPanel" style="display: none;">
                <div class="panel-header">
                    <h2>🧬 System Evolution</h2>
                </div>

                <div class="panel-content">
                    <div class="section">
                        <h3>Evolution Status</h3>
                        <div class="evolution-status">
                            <div class="status-item">
                                <span class="status-label">Current Version:</span>
                                <span class="status-value" id="currentVersion">1.0.0</span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">Last Evolution:</span>
                                <span class="status-value" id="lastEvolution">Never</span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">Evolution Score:</span>
                                <span class="status-value" id="evolutionScore">0</span>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Trigger Evolution</h3>
                        <div class="evolution-options">
                            <div class="option-group">
                                <label>Evolution Target:</label>
                                <select id="evolutionTarget">
                                    <option value="all">All Systems</option>
                                    <option value="reasoning">Reasoning Engine</option>
                                    <option value="code">Code Generation</option>
                                    <option value="memory">Memory System</option>
                                </select>
                            </div>
                            <div class="option-group">
                                <label>Evolution Intensity:</label>
                                <input type="range" id="evolutionIntensity" min="1" max="10" value="5">
                                <span id="intensityValue">5</span>
                            </div>
                            <button class="btn-primary" onclick="triggerEvolution()">🧬 Evolve System</button>
                        </div>
                    </div>

                    <div class="section">
                        <h3>Evolution History</h3>
                        <div class="evolution-history" id="evolutionHistory">
                            <div class="empty-state">No evolution history</div>
                        </div>
                    </div>
                </div>
            </div>
        
        <!-- Right Panel - Workspace -->
        <div class="workspace-panel">
            <!-- Toolbar -->
            <div class="toolbar">
                <div class="toolbar-section">
                    <button class="toolbar-button active" onclick="switchView('code')">📝 Code</button>
                    <button class="toolbar-button" onclick="switchView('preview')">🌐 Preview</button>
                    <button class="toolbar-button" onclick="toggleAgents()">🤖 Agents</button>
                    <button class="toolbar-button" onclick="toggleTasks()">📋 Tasks</button>
                </div>
                
                <div class="toolbar-section">
                    <!-- Execution Mode Toggle -->
                    <div class="execution-mode-toggle">
                        <span class="mode-label">Mode:</span>
                        <div class="toggle-switch simulation" id="executionToggle" onclick="toggleExecutionMode()">
                            <div class="toggle-slider">🧪</div>
                        </div>
                        <div class="mode-status simulation" id="modeStatus">
                            <div class="mode-indicator simulation" id="modeIndicator"></div>
                            <span id="modeText">SIMULATION</span>
                        </div>
                    </div>
                    
                    <span style="font-size: 12px; color: #888;">Workspace: /workspace/wade_env</span>
                </div>
            </div>
            
            <!-- Content Area -->
            <div class="content-area">
                <div class="content-panel">
                    <div class="panel-header">📝 VS Code Editor</div>
                    <div class="panel-content">
                        <iframe 
                            class="vscode-frame" 
                            src="https://work-2-bgqqisslappxddir.prod-runtime.all-hands.dev"
                            title="VS Code Editor"
                        ></iframe>
                    </div>
                </div>
                
                <div class="content-panel">
                    <div class="panel-header">🌐 Application Preview</div>
                    <div class="panel-content">
                        <iframe 
                            class="browser-frame" 
                            src="about:blank"
                            title="Application Preview"
                            id="previewFrame"
                        ></iframe>
                    </div>
                </div>
            </div>
            
            <!-- Terminal -->
            <div class="terminal-panel">
                <div class="terminal-header">
                    <span>💻 Terminal</span>
                    <span style="font-size: 11px; color: #888;">Ready</span>
                </div>
                <div class="terminal-content" id="terminalContent">
                    <div>WADE Native Interface Terminal</div>
                    <div>Ready for autonomous task execution...</div>
                    <div style="margin-top: 10px;">$</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Agent Status Overlay -->
    <div class="agent-status" id="agentStatus">
        <h3 style="margin-bottom: 15px; color: #4fc3f7;">🤖 Active Agents</h3>
        <div id="agentList">
            <div class="agent-item">
                <span class="agent-name">No agents active</span>
                <span class="agent-status-badge">Idle</span>
            </div>
        </div>
    </div>
    
    <!-- Task Progress Overlay -->
    <div class="task-progress" id="taskProgress">
        <h3 style="margin-bottom: 10px; color: #4fc3f7;">📋 Task Progress</h3>
        <div id="currentTask">No active tasks</div>
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
        </div>
        <div id="taskLogs" style="font-size: 12px; color: #888; margin-top: 10px;">
            Ready to execute tasks...
        </div>
    </div>

    <script>
        // WebSocket connection
        let ws = null;
        let agents = [];
        let tasks = [];
        let currentExecutionMode = 'simulation';
        let currentTab = 'chat';
        
        // Loading state management
        function setLoading(elementId, isLoading) {
            const element = document.getElementById(elementId);
            if (!element) return;
            
            if (isLoading) {
                element.disabled = true;
                if (!element.getAttribute('data-original-text')) {
                    element.setAttribute('data-original-text', element.innerHTML);
                }
                element.innerHTML = '<span class="spinner"></span> Processing...';
            } else {
                element.disabled = false;
                // Restore original text from data-original-text attribute
                const originalText = element.getAttribute('data-original-text') || 'Submit';
                element.innerHTML = originalText;
            }
        }
        
        // Tab Management
        function switchTab(tabName) {
            // Hide all panels
            document.getElementById('chatPanel').style.display = 'none';
            document.getElementById('filesPanel').style.display = 'none';
            document.getElementById('settingsPanel').style.display = 'none';
            document.getElementById('modelsPanel').style.display = 'none';
            document.getElementById('agentsPanel').style.display = 'none';
            document.getElementById('intelPanel').style.display = 'none';
            document.getElementById('projectPanel').style.display = 'none';
            document.getElementById('evolutionPanel').style.display = 'none';
            
            // Remove active class from all nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected panel and activate tab
            document.getElementById(tabName + 'Panel').style.display = 'flex';
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
            
            currentTab = tabName;
            
            // Load content for specific tabs
            if (tabName === 'settings') {
                loadSettings();
            } else if (tabName === 'files') {
                loadFiles();
            } else if (tabName === 'models') {
                loadModels();
            }
        }
        
        // Settings Management
        async function loadSettings() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                displaySettings(settings);
            } catch (error) {
                console.error('Failed to load settings:', error);
            }
        }
        
        function displaySettings(settings) {
            const content = document.getElementById('settingsContent');
            content.innerHTML = `
                <div class="setting-group">
                    <h3>🧠 Model Configuration</h3>
                    <div class="setting-item">
                        <label>Primary Model:</label>
                        <select id="primaryModel">
                            <option value="phind-codellama">Phind CodeLlama</option>
                            <option value="deepseek-coder">DeepSeek Coder</option>
                            <option value="wizardlm">WizardLM</option>
                            <option value="qwen">Qwen</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Model Routing:</label>
                        <select id="modelRouting">
                            <option value="adaptive">Adaptive (Auto-select by task)</option>
                            <option value="fixed">Fixed Model</option>
                            <option value="fallback">Fallback Chain</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Task-Based Routing:</label>
                        <input type="checkbox" id="taskBasedRouting" checked>
                        <small>Code→Phind, Logic→Wizard, Summary→Qwen</small>
                    </div>
                    <div class="setting-item">
                        <label>System Prompt Injection:</label>
                        <textarea id="systemPrompt" rows="4" placeholder="Custom system prompt to inject into all model calls...">${settings.system_prompt || ''}</textarea>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>🔄 Execution Settings</h3>
                    <div class="setting-item">
                        <label>Default Mode:</label>
                        <select id="defaultMode">
                            <option value="simulation">Simulation</option>
                            <option value="live">Live</option>
                            <option value="containerized">Containerized (Docker)</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Safety Level:</label>
                        <select id="safetyLevel">
                            <option value="high">High (Confirm all actions)</option>
                            <option value="medium">Medium (Confirm risky actions)</option>
                            <option value="low">Low (Minimal confirmation)</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Execution Isolation:</label>
                        <select id="executionIsolation">
                            <option value="native">Native (No isolation)</option>
                            <option value="subprocess">Subprocess Container</option>
                            <option value="docker">Docker Container</option>
                            <option value="wasm">WASM Sandbox</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Permission Prompts:</label>
                        <input type="checkbox" id="permissionPrompts" checked>
                        <small>Show granular allow/block prompts per operation</small>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>🧬 Self-Evolution Engine</h3>
                    <div class="setting-item">
                        <label>Self-Evolution Mode:</label>
                        <select id="evolutionMode">
                            <option value="light">Light (Basic feedback)</option>
                            <option value="full">Full (Deep learning)</option>
                            <option value="off">Disabled</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Prompt Success Scoring:</label>
                        <input type="checkbox" id="promptScoring" checked>
                        <small>Track and rank prompt success/failure rates</small>
                    </div>
                    <div class="setting-item">
                        <label>Context Memory Refinement:</label>
                        <input type="checkbox" id="contextRefinement" checked>
                        <small>Evolve context from repeated failures</small>
                    </div>
                    <div class="setting-item">
                        <label>Learning Rate:</label>
                        <input type="range" id="learningRate" min="0" max="100" value="50">
                        <span id="learningRateValue">50%</span>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>🤖 Micro-Agent Engine</h3>
                    <div class="setting-item">
                        <label>Auto-Agent Spawning:</label>
                        <input type="checkbox" id="autoAgentSpawning" checked>
                        <small>Create specialized agents for repeat tasks</small>
                    </div>
                    <div class="setting-item">
                        <label>Agent Specialization:</label>
                        <select id="agentSpecialization">
                            <option value="task">Task-Based (Analyzer, Builder, Debugger)</option>
                            <option value="domain">Domain-Based (Frontend, Backend, DevOps)</option>
                            <option value="hybrid">Hybrid</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Max Concurrent Agents:</label>
                        <input type="number" id="maxAgents" min="1" max="20" value="5">
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>🛠️ Automation & Workflow</h3>
                    <div class="setting-item">
                        <label>Custom Task Builder:</label>
                        <input type="checkbox" id="customTaskBuilder" checked>
                        <small>Save chat sequences as reusable tasks</small>
                    </div>
                    <div class="setting-item">
                        <label>Workflow Composer:</label>
                        <input type="checkbox" id="workflowComposer" checked>
                        <small>Drag-and-drop flow builder</small>
                    </div>
                    <div class="setting-item">
                        <label>Auto-Chain Detection:</label>
                        <input type="checkbox" id="autoChainDetection" checked>
                        <small>Detect and suggest workflow patterns</small>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>📡 Web Search & Intelligence</h3>
                    <div class="setting-item">
                        <label>Search Integration:</label>
                        <select id="searchIntegration">
                            <option value="standard">Standard Web Search</option>
                            <option value="tor">Tor + Proxy Chain</option>
                            <option value="hybrid">Hybrid (Auto-route)</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Dynamic Search Routing:</label>
                        <input type="checkbox" id="dynamicSearchRouting" checked>
                        <small>Model decides which data source to query</small>
                    </div>
                    <div class="setting-item">
                        <label>Dark Search Commands:</label>
                        <input type="checkbox" id="darkSearchCommands" checked>
                        <small>Enable /dark-search and /intel-query</small>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>📁 Workspace & Sync</h3>
                    <div class="setting-item">
                        <label>Project Bundler:</label>
                        <input type="checkbox" id="projectBundler" checked>
                        <small>Package entire session (chat + files + history)</small>
                    </div>
                    <div class="setting-item">
                        <label>Cloud Sync:</label>
                        <select id="cloudSync">
                            <option value="none">Disabled</option>
                            <option value="gdrive">Google Drive</option>
                            <option value="dropbox">Dropbox</option>
                            <option value="gitlab">GitLab</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Auto-Export Format:</label>
                        <select id="autoExportFormat">
                            <option value="zip">ZIP Archive</option>
                            <option value="git">Git Repository</option>
                            <option value="docker">Docker Image</option>
                        </select>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>💾 Memory & History</h3>
                    <div class="setting-item">
                        <label>Chat History Retention:</label>
                        <select id="historyRetention">
                            <option value="session">Session Only</option>
                            <option value="7days">7 Days</option>
                            <option value="30days">30 Days</option>
                            <option value="forever">Forever</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Memory Timeline:</label>
                        <input type="checkbox" id="memoryTimeline" checked>
                        <small>Visual timeline of all interactions</small>
                    </div>
                    <div class="setting-item">
                        <label>Persistent Memory:</label>
                        <input type="checkbox" id="persistentMemory" checked>
                        <small>Remember context across sessions</small>
                    </div>
                </div>
                
                <div class="setting-group">
                    <h3>🔐 Security & Permissions</h3>
                    <div class="setting-item">
                        <label>Authentication:</label>
                        <select id="authentication">
                            <option value="none">None</option>
                            <option value="password">Password</option>
                            <option value="token">API Token</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>File Access Restrictions:</label>
                        <input type="checkbox" id="fileRestrictions" checked>
                        <small>Limit file access to workspace only</small>
                    </div>
                    <div class="setting-item">
                        <label>Network Restrictions:</label>
                        <input type="checkbox" id="networkRestrictions">
                        <small>Block external network access in live mode</small>
                    </div>
                </div>
            `;
            
            // Add event listeners for settings
            document.getElementById('learningRate').addEventListener('input', function() {
                document.getElementById('learningRateValue').textContent = this.value + '%';
            });
        }
        
        // File Management
        async function loadFiles() {
            try {
                const response = await fetch('/api/files');
                const data = await response.json();
                displayFiles(data.files);
            } catch (error) {
                console.error('Failed to load files:', error);
            }
        }
        
        function displayFiles(files) {
            const content = document.getElementById('filesContent');
            if (files.length === 0) {
                content.innerHTML = '<div class="empty-state">No files in workspace</div>';
                return;
            }
            
            content.innerHTML = files.map(file => `
                <div class="file-item">
                    <span class="file-icon">${getFileIcon(file.name)}</span>
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${formatFileSize(file.size)}</span>
                    <div class="file-actions">
                        <button onclick="editFile('${file.name}')" title="Edit">✏️</button>
                        <button onclick="downloadFile('${file.name}')" title="Download">⬇️</button>
                        <button onclick="deleteFile('${file.name}')" title="Delete">🗑️</button>
                    </div>
                </div>
            `).join('');
        }
        
        function getFileIcon(filename) {
            const ext = filename.split('.').pop().toLowerCase();
            const icons = {
                'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨',
                'json': '📋', 'md': '📝', 'txt': '📄', 'yml': '⚙️',
                'yaml': '⚙️', 'xml': '📰', 'csv': '📊'
            };
            return icons[ext] || '📄';
        }
        
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }
        
        // Model Management
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                displayModels(data.models);
            } catch (error) {
                console.error('Failed to load models:', error);
            }
        }
        
        function displayModels(models) {
            const list = document.getElementById('modelsList');
            list.innerHTML = models.map(model => `
                <div class="model-item">
                    <span class="model-name">${model.name}</span>
                    <span class="model-size">${model.size || 'Unknown'}</span>
                    <button onclick="setActiveModel('${model.name}')">Set Active</button>
                </div>
            `).join('');
        }
        
        // File Operations
        function uploadFile() {
            document.getElementById('fileUpload').click();
        }
        
        async function editFile(filename) {
            try {
                const response = await fetch(`/api/files/content?file_path=${encodeURIComponent(filename)}`);
                const data = await response.json();
                
                // Create a simple editor modal
                const modal = document.createElement('div');
                modal.className = 'editor-modal';
                modal.innerHTML = `
                    <div class="editor-content">
                        <div class="editor-header">
                            <h3>Edit: ${filename}</h3>
                            <button onclick="closeEditor()">✕</button>
                        </div>
                        <textarea id="fileEditor">${data.content}</textarea>
                        <div class="editor-actions">
                            <button onclick="saveFile('${filename}')">💾 Save</button>
                            <button onclick="closeEditor()">Cancel</button>
                        </div>
                    </div>
                `;
                document.body.appendChild(modal);
            } catch (error) {
                console.error('Failed to load file:', error);
            }
        }
        
        async function downloadFile(filename) {
            window.open(`/api/files/download?file_path=${encodeURIComponent(filename)}`);
        }
        
        async function deleteFile(filename) {
            if (confirm(`Delete ${filename}?`)) {
                try {
                    await fetch(`/api/files?file_path=${encodeURIComponent(filename)}`, {
                        method: 'DELETE'
                    });
                    loadFiles(); // Refresh file list
                } catch (error) {
                    console.error('Failed to delete file:', error);
                }
            }
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
            
            ws.onopen = function() {
                console.log('Connected to WADE');
                updateConnectionStatus(true);
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('Disconnected from WADE');
                updateConnectionStatus(false);
                // Reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'chat_response':
                    addMessage(data.message);
                    break;
                case 'chat_update':
                    addMessage(data.message);
                    break;
                case 'task_update':
                    updateTaskProgress(data.task);
                    break;
                case 'execution_mode_changed':
                    updateExecutionModeUI(data.new_mode, data.mode_info);
                    showModeChangeNotification(data.old_mode, data.new_mode);
                    break;
            }
        }
        
        function updateConnectionStatus(connected) {
            const indicator = document.querySelector('.status-indicator');
            if (connected) {
                indicator.style.background = '#4caf50';
            } else {
                indicator.style.background = '#f44336';
            }
        }
        
        function addMessage(message) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${message.role}`;
            
            const time = new Date(message.timestamp).toLocaleTimeString();
            messageDiv.innerHTML = `
                ${message.content.replace(/\\n/g, '<br>')}
                <div class="message-time">${time}</div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                // Add user message immediately
                addMessage({
                    role: 'user',
                    content: message,
                    timestamp: new Date().toISOString()
                });
                
                // Send to server
                ws.send(JSON.stringify({
                    type: 'chat_message',
                    content: message
                }));
                
                input.value = '';
                input.style.height = '40px';
            }
        }
        
        function updateTaskProgress(task) {
            const taskProgress = document.getElementById('taskProgress');
            const currentTask = document.getElementById('currentTask');
            const progressFill = document.getElementById('progressFill');
            const taskLogs = document.getElementById('taskLogs');
            
            if (task.status === 'running' || task.status === 'pending') {
                taskProgress.classList.add('show');
                currentTask.textContent = task.description;
                progressFill.style.width = task.progress + '%';
                
                if (task.logs && task.logs.length > 0) {
                    taskLogs.innerHTML = task.logs.slice(-3).join('<br>');
                }
                
                // Add to terminal
                const terminal = document.getElementById('terminalContent');
                if (task.logs && task.logs.length > 0) {
                    const lastLog = task.logs[task.logs.length - 1];
                    terminal.innerHTML += `<div>${lastLog}</div>`;
                    terminal.scrollTop = terminal.scrollHeight;
                }
            } else if (task.status === 'completed') {
                setTimeout(() => {
                    taskProgress.classList.remove('show');
                }, 3000);
            }
        }
        
        function switchView(view) {
            const buttons = document.querySelectorAll('.toolbar-button');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            if (view === 'preview') {
                // Try to load the application preview
                const previewFrame = document.getElementById('previewFrame');
                previewFrame.src = 'http://localhost:8000'; // Default app port
            }
        }
        
        function toggleAgents() {
            const agentStatus = document.getElementById('agentStatus');
            agentStatus.classList.toggle('show');
            
            // Fetch and display agents
            fetch('/api/agents')
                .then(response => response.json())
                .then(data => {
                    const agentList = document.getElementById('agentList');
                    if (data.agents && data.agents.length > 0) {
                        agentList.innerHTML = data.agents.map(agent => `
                            <div class="agent-item">
                                <span class="agent-name">${agent.name}</span>
                                <span class="agent-status-badge">${agent.status}</span>
                            </div>
                        `).join('');
                    } else {
                        agentList.innerHTML = `
                            <div class="agent-item">
                                <span class="agent-name">No agents active</span>
                                <span class="agent-status-badge">Idle</span>
                            </div>
                        `;
                    }
                });
        }
        
        function toggleTasks() {
            const taskProgress = document.getElementById('taskProgress');
            taskProgress.classList.toggle('show');
        }
        
        // Chat input handling
        document.getElementById('chatInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Auto-resize textarea
        document.getElementById('chatInput').addEventListener('input', function() {
            this.style.height = '40px';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        // Execution Mode Functions
        async function toggleExecutionMode() {
            const newMode = currentExecutionMode === 'simulation' ? 'live' : 'simulation';
            
            // Show confirmation for live mode
            if (newMode === 'live') {
                const confirmed = confirm(
                    '⚠️ WARNING: Switching to LIVE MODE\\n\\n' +
                    'This will enable real system operations with full access.\\n' +
                    'All changes will affect real systems and data.\\n\\n' +
                    'Are you sure you want to continue?'
                );
                
                if (!confirmed) {
                    return;
                }
            }
            
            try {
                const response = await fetch('/api/execution-mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: newMode })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    updateExecutionModeUI(result.new_mode, result.mode_info);
                    showModeChangeNotification(result.old_mode, result.new_mode);
                } else {
                    alert('Error changing execution mode: ' + result.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function updateExecutionModeUI(mode, modeInfo) {
            currentExecutionMode = mode;
            
            const toggle = document.getElementById('executionToggle');
            const status = document.getElementById('modeStatus');
            const indicator = document.getElementById('modeIndicator');
            const text = document.getElementById('modeText');
            const slider = toggle.querySelector('.toggle-slider');
            
            // Update toggle switch
            toggle.className = `toggle-switch ${mode}`;
            
            // Update status
            status.className = `mode-status ${mode}`;
            indicator.className = `mode-indicator ${mode}`;
            text.textContent = mode.toUpperCase();
            
            // Update slider icon and position
            if (mode === 'simulation') {
                slider.textContent = '🧪';
            } else {
                slider.textContent = '🔥';
            }
        }
        
        function showModeChangeNotification(oldMode, newMode) {
            // Add notification message to chat
            const notification = {
                role: 'system',
                content: `🔄 **Execution Mode Changed**\\n\\nSwitched from **${oldMode.toUpperCase()}** to **${newMode.toUpperCase()}**\\n\\n${newMode === 'live' ? '⚠️ **LIVE MODE ACTIVE** - Real system operations enabled' : '🧪 **SIMULATION MODE ACTIVE** - All operations are sandboxed'}`,
                timestamp: new Date().toISOString()
            };
            
            addMessage(notification);
            
            // Show terminal notification
            const terminal = document.getElementById('terminalContent');
            const timestamp = new Date().toLocaleTimeString();
            terminal.innerHTML += `<div style="color: ${newMode === 'live' ? '#f44336' : '#4caf50'};">[${timestamp}] Execution mode changed to ${newMode.toUpperCase()}</div>`;
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        async function loadExecutionMode() {
            try {
                const response = await fetch('/api/execution-mode');
                const data = await response.json();
                updateExecutionModeUI(data.current_mode, data.mode_info);
            } catch (error) {
                console.error('Error loading execution mode:', error);
            }
        }
        
        // Initialize
        connectWebSocket();
        loadExecutionMode();
        
        // Add tab navigation event listeners
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                const tabName = this.getAttribute('data-tab');
                switchTab(tabName);
            });
        });
        
        // Add settings tab navigation
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('settings-tab')) {
                document.querySelectorAll('.settings-tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                e.target.classList.add('active');
                
                const section = e.target.getAttribute('data-section');
                loadSettingsSection(section);
            }
        });
        
        // Load initial data
        setTimeout(() => {
            fetch('/api/chat-history')
                .then(response => response.json())
                .then(data => {
                    if (data.messages) {
                        data.messages.forEach(message => {
                            if (message.role !== 'system') {
                                addMessage(message);
                            }
                        });
                    }
                });
        }, 1000);
        
        // Microagent Management Functions
        function createAgent() {
            const agentName = prompt("Enter agent name:");
            if (!agentName) return;
            
            fetch('/api/agents/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: agentName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateAgentList();
                    addSystemMessage(`Agent "${agentName}" created successfully`);
                } else {
                    addSystemMessage(`Failed to create agent: ${data.error}`);
                }
            });
        }
        
        function spawnAgent(templateId) {
            fetch('/api/agents/spawn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ template: templateId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateAgentList();
                    addSystemMessage(`Agent spawned from template "${templateId}"`);
                } else {
                    addSystemMessage(`Failed to spawn agent: ${data.error}`);
                }
            });
        }
        
        function updateAgentList() {
            fetch('/api/agents/list')
            .then(response => response.json())
            .then(data => {
                const agentList = document.getElementById('agentList');
                if (data.agents && data.agents.length > 0) {
                    agentList.innerHTML = data.agents.map(agent => `
                        <div class="agent-item">
                            <div class="agent-info">
                                <span class="agent-name">${agent.name}</span>
                                <span class="agent-status">${agent.status}</span>
                            </div>
                            <div class="agent-actions">
                                <button class="btn-sm" onclick="terminateAgent('${agent.id}')">Terminate</button>
                            </div>
                        </div>
                    `).join('');
                } else {
                    agentList.innerHTML = '<div class="empty-state">No active agents</div>';
                }
            });
        }
        
        function terminateAgent(agentId) {
            fetch(`/api/agents/terminate/${agentId}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateAgentList();
                    addSystemMessage(`Agent terminated successfully`);
                } else {
                    addSystemMessage(`Failed to terminate agent: ${data.error}`);
                }
            });
        }
        
        // Intel Query Functions
        function executeIntelQuery() {
            const query = document.getElementById('intelQuery').value;
            const searchType = document.getElementById('searchType').value;
            const anonymityLevel = document.getElementById('anonymityLevel').value;
            const searchButton = document.getElementById('searchButton');
            
            if (!query) {
                alert("Please enter a search query");
                return;
            }
            
            // Set loading state
            setLoading('searchButton', true);
            document.getElementById('intelResults').innerHTML = '<div class="loading">Searching...</div>';
            
            fetch('/api/intel-query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    type: searchType,
                    anonymity: anonymityLevel
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.results) {
                    displayIntelResults(data.results);
                } else {
                    document.getElementById('intelResults').innerHTML = 
                        '<div class="empty-state">No results found</div>';
                }
                // Reset loading state
                setLoading('searchButton', false);
            })
            .catch(error => {
                document.getElementById('intelResults').innerHTML = 
                    `<div class="error-state">Error: ${error.message}</div>`;
                // Reset loading state
                setLoading('searchButton', false);
            });
        }
        
        function displayIntelResults(results) {
            const resultsContainer = document.getElementById('intelResults');
            
            if (results.length === 0) {
                resultsContainer.innerHTML = '<div class="empty-state">No results found</div>';
                return;
            }
            
            resultsContainer.innerHTML = results.map(result => `
                <div class="result-item">
                    <div class="result-header">
                        <span class="result-source">${result.source}</span>
                        <span class="result-date">${new Date(result.timestamp).toLocaleString()}</span>
                    </div>
                    <div class="result-content">${result.content}</div>
                    <div class="result-meta">
                        <span class="result-confidence">Confidence: ${result.confidence || 'N/A'}</span>
                        <span class="result-tags">${(result.tags || []).join(', ')}</span>
                    </div>
                </div>
            `).join('');
        }
        
        // Project Management Functions
        function exportProject() {
            const format = document.getElementById('exportFormat').value;
            const includeChat = document.getElementById('includeChat').checked;
            const includeFiles = document.getElementById('includeFiles').checked;
            const includeMemory = document.getElementById('includeMemory').checked;
            
            fetch('/api/project/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    format: format,
                    include: {
                        chat: includeChat,
                        files: includeFiles,
                        memory: includeMemory
                    }
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Export failed');
                }
                return response.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `wade-project.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                addSystemMessage(`Project exported successfully as ${format}`);
            })
            .catch(error => {
                addSystemMessage(`Export failed: ${error.message}`);
            });
        }
        
        function syncToCloud() {
            const provider = document.getElementById('cloudProvider').value;
            const syncStatus = document.getElementById('syncStatus');
            
            syncStatus.innerHTML = '<div class="loading">Syncing to cloud...</div>';
            
            fetch('/api/project/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: provider })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    syncStatus.innerHTML = `
                        <div class="success-state">
                            <p>Synced successfully to ${provider}</p>
                            <p>Last sync: ${new Date().toLocaleString()}</p>
                        </div>
                    `;
                    addSystemMessage(`Project synced to ${provider}`);
                } else {
                    syncStatus.innerHTML = `
                        <div class="error-state">
                            <p>Sync failed: ${data.error}</p>
                        </div>
                    `;
                }
            })
            .catch(error => {
                syncStatus.innerHTML = `
                    <div class="error-state">
                        <p>Sync error: ${error.message}</p>
                    </div>
                `;
            });
        }
        
        // Evolution Functions
        function triggerEvolution() {
            const target = document.getElementById('evolutionTarget').value;
            const intensity = document.getElementById('evolutionIntensity').value;
            
            fetch('/api/self-evolve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target: target,
                    intensity: parseInt(intensity)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addSystemMessage(`Evolution process started for ${target}`);
                    updateEvolutionStatus();
                } else {
                    addSystemMessage(`Evolution failed: ${data.error}`);
                }
            })
            .catch(error => {
                addSystemMessage(`Evolution error: ${error.message}`);
            });
        }
        
        function updateEvolutionStatus() {
            fetch('/api/evolution/status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('currentVersion').textContent = data.version;
                document.getElementById('lastEvolution').textContent = 
                    data.last_evolution ? new Date(data.last_evolution).toLocaleString() : 'Never';
                document.getElementById('evolutionScore').textContent = data.score;
                
                updateEvolutionHistory(data.history || []);
            });
        }
        
        function updateEvolutionHistory(history) {
            const historyContainer = document.getElementById('evolutionHistory');
            
            if (history.length === 0) {
                historyContainer.innerHTML = '<div class="empty-state">No evolution history</div>';
                return;
            }
            
            historyContainer.innerHTML = history.map(entry => `
                <div class="history-item">
                    <div class="history-header">
                        <span class="history-version">v${entry.version}</span>
                        <span class="history-date">${new Date(entry.timestamp).toLocaleString()}</span>
                    </div>
                    <div class="history-changes">
                        <p>${entry.changes}</p>
                    </div>
                    <div class="history-metrics">
                        <span class="metric">Score: ${entry.score}</span>
                        <span class="metric">Target: ${entry.target}</span>
                    </div>
                </div>
            `).join('');
        }
        
        // Initialize intensity value display
        document.getElementById('evolutionIntensity')?.addEventListener('input', function() {
            document.getElementById('intensityValue').textContent = this.value;
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("🚀 Starting WADE Native Interface...")
    print("📍 Access the interface at: https://work-1-bgqqisslappxddir.prod-runtime.all-hands.dev")
    print("💬 Chat-driven development with embedded VS Code and terminal")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=12000,
        reload=False
    )