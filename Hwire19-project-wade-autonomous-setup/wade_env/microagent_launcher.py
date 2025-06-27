#!/usr/bin/env python3
"""
WADE Microagent Launcher - Autonomous task-specific agent management
Spawns and manages specialized microagents for different tasks
"""

import os
import json
import logging
import asyncio
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import re

# Import WADE components
try:
    from settings_manager import settings_manager
    from model_router import model_router
except ImportError:
    # For standalone testing
    from wade_env.settings_manager import settings_manager
    from wade_env.model_router import model_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("microagent_launcher")

class AgentStatus(Enum):
    """Status of a microagent"""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"

class AgentType(Enum):
    """Types of microagents"""
    GENERAL = "general"
    CODE = "code"
    SECURITY = "security"
    DATA = "data"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    BACKEND = "backend"
    TESTING = "testing"
    RESEARCH = "research"

class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class MicroagentConfig:
    """Configuration for a microagent"""
    agent_type: AgentType
    name: str
    description: str
    capabilities: List[str]
    triggers: List[str]
    model_preference: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)

@dataclass
class Microagent:
    """Microagent data structure"""
    agent_id: str
    config: MicroagentConfig
    status: AgentStatus
    created_at: float
    updated_at: float
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task_id: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Task:
    """Task data structure"""
    task_id: str
    agent_id: str
    description: str
    status: TaskStatus
    created_at: float
    updated_at: float
    priority: int = 3  # 1-5, with 5 being highest
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "simulation"  # "simulation" or "live"

class MicroagentLauncher:
    """
    Microagent launcher and manager
    Spawns and manages specialized microagents for different tasks
    """
    
    def __init__(self):
        """Initialize the microagent launcher"""
        self.settings = self._load_launcher_settings()
        self.data_dir = self.settings.get("data_dir", "/workspace/wade_env/microagents")
        self.agents: Dict[str, Microagent] = {}
        self.tasks: Dict[str, Task] = {}
        self.agent_configs: Dict[AgentType, MicroagentConfig] = self._load_agent_configs()
        self.max_concurrent_agents = self.settings.get("max_concurrent_agents", 5)
        self.default_execution_mode = self.settings.get("default_execution_mode", "simulation")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing data
        self._load_data()
    
    def _load_launcher_settings(self) -> Dict[str, Any]:
        """Load launcher settings from settings manager"""
        try:
            launcher_settings = settings_manager.get_settings_dict().get("microagent_launcher", {})
            if not launcher_settings:
                # Initialize with defaults if not present
                launcher_settings = {
                    "enabled": True,
                    "data_dir": "/workspace/wade_env/microagents",
                    "max_concurrent_agents": 5,
                    "default_execution_mode": "simulation",
                    "auto_spawn": True,
                    "agent_timeout_seconds": 300,
                    "task_timeout_seconds": 600
                }
                settings_manager.update_settings("microagent_launcher", launcher_settings)
            return launcher_settings
        except Exception as e:
            logger.error(f"Error loading launcher settings: {e}")
            return {
                "enabled": True,
                "data_dir": "/workspace/wade_env/microagents",
                "max_concurrent_agents": 5,
                "default_execution_mode": "simulation",
                "auto_spawn": True,
                "agent_timeout_seconds": 300,
                "task_timeout_seconds": 600
            }
    
    def _load_agent_configs(self) -> Dict[AgentType, MicroagentConfig]:
        """Load agent configurations"""
        configs = {}
        
        # Default configurations for different agent types
        configs[AgentType.GENERAL] = MicroagentConfig(
            agent_type=AgentType.GENERAL,
            name="General Purpose Agent",
            description="Versatile agent for general tasks and coordination",
            capabilities=["Task planning", "Information retrieval", "Basic code understanding", "Coordination"],
            triggers=["help", "general", "assistant", "guide"],
            model_preference="gpt-4o",
            system_prompt="You are a versatile general-purpose assistant that can help with a wide range of tasks."
        )
        
        configs[AgentType.CODE] = MicroagentConfig(
            agent_type=AgentType.CODE,
            name="Code Specialist Agent",
            description="Specialized agent for code generation and software development",
            capabilities=["Code generation", "Debugging", "Refactoring", "Code review", "Documentation"],
            triggers=["code", "program", "develop", "implement", "debug", "function", "class"],
            model_preference="phind-codellama",
            system_prompt="You are a code specialist focused on writing clean, efficient, and well-documented code."
        )
        
        configs[AgentType.SECURITY] = MicroagentConfig(
            agent_type=AgentType.SECURITY,
            name="Security Specialist Agent",
            description="Specialized agent for security analysis and hardening",
            capabilities=["Vulnerability assessment", "Security review", "Penetration testing", "Hardening"],
            triggers=["security", "vulnerability", "hack", "exploit", "secure", "penetration", "audit"],
            model_preference="claude-3-opus",
            system_prompt="You are a security specialist focused on identifying and mitigating security vulnerabilities."
        )
        
        configs[AgentType.DATA] = MicroagentConfig(
            agent_type=AgentType.DATA,
            name="Data Analysis Agent",
            description="Specialized agent for data processing and analysis",
            capabilities=["Data cleaning", "Statistical analysis", "Visualization", "Machine learning"],
            triggers=["data", "analyze", "statistics", "dataset", "visualization", "plot", "chart"],
            model_preference="qwen-72b",
            system_prompt="You are a data analysis specialist focused on extracting insights from data."
        )
        
        configs[AgentType.DEVOPS] = MicroagentConfig(
            agent_type=AgentType.DEVOPS,
            name="DevOps Agent",
            description="Specialized agent for infrastructure and deployment",
            capabilities=["Infrastructure as Code", "CI/CD", "Containerization", "Orchestration"],
            triggers=["devops", "deploy", "infrastructure", "docker", "kubernetes", "ci/cd", "pipeline"],
            model_preference="wizard-mega",
            system_prompt="You are a DevOps specialist focused on infrastructure, deployment, and automation."
        )
        
        configs[AgentType.FRONTEND] = MicroagentConfig(
            agent_type=AgentType.FRONTEND,
            name="Frontend Development Agent",
            description="Specialized agent for frontend development",
            capabilities=["HTML/CSS", "JavaScript", "UI/UX", "Responsive design", "Frontend frameworks"],
            triggers=["frontend", "ui", "interface", "web", "html", "css", "javascript", "react", "vue", "angular"],
            model_preference="phind-codellama",
            system_prompt="You are a frontend development specialist focused on creating beautiful and functional user interfaces."
        )
        
        configs[AgentType.BACKEND] = MicroagentConfig(
            agent_type=AgentType.BACKEND,
            name="Backend Development Agent",
            description="Specialized agent for backend development",
            capabilities=["API design", "Database", "Authentication", "Server-side logic", "Microservices"],
            triggers=["backend", "server", "api", "database", "auth", "microservice", "endpoint"],
            model_preference="phind-codellama",
            system_prompt="You are a backend development specialist focused on building robust and scalable server-side applications."
        )
        
        configs[AgentType.TESTING] = MicroagentConfig(
            agent_type=AgentType.TESTING,
            name="Testing Agent",
            description="Specialized agent for software testing",
            capabilities=["Unit testing", "Integration testing", "E2E testing", "Test automation", "QA"],
            triggers=["test", "testing", "qa", "quality", "assert", "verify", "validation"],
            model_preference="wizard-mega",
            system_prompt="You are a testing specialist focused on ensuring software quality through comprehensive testing."
        )
        
        configs[AgentType.RESEARCH] = MicroagentConfig(
            agent_type=AgentType.RESEARCH,
            name="Research Agent",
            description="Specialized agent for research and information gathering",
            capabilities=["Literature review", "Information synthesis", "Trend analysis", "Academic research"],
            triggers=["research", "study", "investigate", "analyze", "review", "survey", "literature"],
            model_preference="claude-3-opus",
            system_prompt="You are a research specialist focused on gathering, analyzing, and synthesizing information."
        )
        
        # Override with any custom configurations from settings
        custom_configs = self.settings.get("agent_configs", {})
        for agent_type_str, config_data in custom_configs.items():
            try:
                agent_type = AgentType(agent_type_str)
                if agent_type in configs:
                    # Update existing config
                    for key, value in config_data.items():
                        if hasattr(configs[agent_type], key):
                            setattr(configs[agent_type], key, value)
            except ValueError:
                logger.warning(f"Unknown agent type: {agent_type_str}")
        
        return configs
    
    def _load_data(self) -> None:
        """Load existing data from files"""
        try:
            # Load agents
            agents_path = os.path.join(self.data_dir, "agents.json")
            if os.path.exists(agents_path):
                with open(agents_path, 'r') as f:
                    agents_data = json.load(f)
                    for agent_id, agent_data in agents_data.items():
                        # Convert string enums to actual enums
                        agent_data["status"] = AgentStatus(agent_data["status"])
                        agent_data["config"]["agent_type"] = AgentType(agent_data["config"]["agent_type"])
                        self.agents[agent_id] = Microagent(**agent_data)
            
            # Load tasks
            tasks_path = os.path.join(self.data_dir, "tasks.json")
            if os.path.exists(tasks_path):
                with open(tasks_path, 'r') as f:
                    tasks_data = json.load(f)
                    for task_id, task_data in tasks_data.items():
                        # Convert string enum to actual enum
                        task_data["status"] = TaskStatus(task_data["status"])
                        self.tasks[task_id] = Task(**task_data)
            
            logger.info(f"Loaded {len(self.agents)} agents and {len(self.tasks)} tasks")
        except Exception as e:
            logger.error(f"Error loading microagent data: {e}")
    
    def _save_data(self) -> None:
        """Save data to files"""
        try:
            # Save agents
            agents_data = {}
            for agent_id, agent in self.agents.items():
                agent_dict = asdict(agent)
                # Convert enums to strings for JSON serialization
                agent_dict["status"] = agent_dict["status"].value
                agent_dict["config"]["agent_type"] = agent_dict["config"]["agent_type"].value
                agents_data[agent_id] = agent_dict
            
            agents_path = os.path.join(self.data_dir, "agents.json")
            with open(agents_path, 'w') as f:
                json.dump(agents_data, f, indent=2)
            
            # Save tasks
            tasks_data = {}
            for task_id, task in self.tasks.items():
                task_dict = asdict(task)
                # Convert enum to string for JSON serialization
                task_dict["status"] = task_dict["status"].value
                tasks_data[task_id] = task_dict
            
            tasks_path = os.path.join(self.data_dir, "tasks.json")
            with open(tasks_path, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            logger.info("Microagent data saved successfully")
        except Exception as e:
            logger.error(f"Error saving microagent data: {e}")
    
    def identify_agent_type(self, message: str) -> Tuple[AgentType, float]:
        """
        Identify the most appropriate agent type for a message
        Returns the agent type and a confidence score
        """
        message_lower = message.lower()
        
        # Check for explicit agent type mentions
        for agent_type in AgentType:
            if agent_type.value in message_lower:
                return agent_type, 0.9
        
        # Check for triggers in each agent config
        type_scores = {}
        for agent_type, config in self.agent_configs.items():
            score = 0
            for trigger in config.triggers:
                if trigger in message_lower:
                    score += 1
            
            if score > 0:
                # Normalize score based on number of triggers
                normalized_score = min(0.9, score / len(config.triggers) + 0.3)
                type_scores[agent_type] = normalized_score
        
        if type_scores:
            # Return the agent type with the highest score
            best_type = max(type_scores.items(), key=lambda x: x[1])
            return best_type[0], best_type[1]
        
        # Default to general agent with low confidence
        return AgentType.GENERAL, 0.5
    
    def is_task_like(self, message: str) -> bool:
        """
        Determine if a message is task-like (contains an actionable request)
        Returns True if the message appears to be a task
        """
        message_lower = message.lower()
        
        # Task-like indicators
        task_verbs = [
            "create", "make", "build", "develop", "implement", "code",
            "write", "generate", "analyze", "design", "fix", "debug",
            "optimize", "refactor", "test", "deploy", "configure",
            "setup", "install", "find", "search", "check", "review"
        ]
        
        # Check for task verbs
        for verb in task_verbs:
            if re.search(r'\b' + verb + r'\b', message_lower):
                return True
        
        # Check for question-like patterns (not tasks)
        question_patterns = [
            r'^what\b', r'^how\b', r'^why\b', r'^when\b', r'^where\b',
            r'^who\b', r'^which\b', r'^can you explain\b', r'^tell me\b'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, message_lower):
                return False
        
        # Check for imperative sentences
        if re.search(r'^[a-z]+\b', message_lower):  # Starts with a verb
            return True
        
        return False
    
    async def spawn_agent(self, agent_type: Union[AgentType, str], 
                        name: Optional[str] = None) -> str:
        """
        Spawn a new microagent of the specified type
        Returns the ID of the spawned agent
        """
        # Convert string to enum if needed
        if isinstance(agent_type, str):
            try:
                agent_type = AgentType(agent_type)
            except ValueError:
                logger.error(f"Unknown agent type: {agent_type}")
                return None
        
        # Check if we've reached the maximum number of concurrent agents
        active_agents = [a for a in self.agents.values() 
                        if a.status != AgentStatus.TERMINATED]
        
        if len(active_agents) >= self.max_concurrent_agents:
            # Find an idle agent to replace
            idle_agents = [a for a in active_agents if a.status == AgentStatus.IDLE]
            if idle_agents:
                # Terminate the oldest idle agent
                oldest_agent = min(idle_agents, key=lambda a: a.updated_at)
                oldest_agent.status = AgentStatus.TERMINATED
                oldest_agent.updated_at = time.time()
                logger.info(f"Terminated idle agent {oldest_agent.agent_id} to make room for new agent")
            else:
                logger.warning(f"Cannot spawn new agent: maximum concurrent agents reached")
                return None
        
        # Get the configuration for this agent type
        if agent_type not in self.agent_configs:
            logger.error(f"No configuration found for agent type: {agent_type}")
            return None
        
        config = self.agent_configs[agent_type]
        
        # Generate a unique agent ID
        agent_id = f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Create the agent
        agent = Microagent(
            agent_id=agent_id,
            config=config,
            status=AgentStatus.IDLE,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Customize name if provided
        if name:
            agent.config.name = name
        
        # Store the agent
        self.agents[agent_id] = agent
        
        # Save data
        self._save_data()
        
        logger.info(f"Spawned new agent: {agent_id} ({agent.config.name})")
        return agent_id
    
    async def create_task(self, description: str, agent_id: Optional[str] = None,
                        priority: int = 3, context: Optional[Dict[str, Any]] = None,
                        parent_task_id: Optional[str] = None,
                        execution_mode: Optional[str] = None) -> str:
        """
        Create a new task
        Returns the ID of the created task
        """
        # Validate priority
        priority = max(1, min(5, priority))
        
        # Use default execution mode if not specified
        if execution_mode is None:
            execution_mode = self.default_execution_mode
        
        # If no agent ID is provided, identify the appropriate agent type
        if agent_id is None:
            agent_type, confidence = self.identify_agent_type(description)
            
            # Find an existing idle agent of this type
            matching_agents = [a for a in self.agents.values() 
                             if a.config.agent_type == agent_type and a.status == AgentStatus.IDLE]
            
            if matching_agents:
                # Use an existing idle agent
                agent = matching_agents[0]
                agent_id = agent.agent_id
            else:
                # Spawn a new agent
                agent_id = await self.spawn_agent(agent_type)
                
                if agent_id is None:
                    logger.error("Failed to spawn agent for task")
                    return None
        
        # Generate a unique task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create the task
        task = Task(
            task_id=task_id,
            agent_id=agent_id,
            description=description,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            updated_at=time.time(),
            priority=priority,
            context=context or {},
            parent_task_id=parent_task_id,
            execution_mode=execution_mode
        )
        
        # Store the task
        self.tasks[task_id] = task
        
        # If this is a subtask, add it to the parent task
        if parent_task_id and parent_task_id in self.tasks:
            parent_task = self.tasks[parent_task_id]
            parent_task.subtasks.append(task_id)
            parent_task.updated_at = time.time()
        
        # Save data
        self._save_data()
        
        logger.info(f"Created task: {task_id} for agent {agent_id}")
        
        # Start task execution in background
        asyncio.create_task(self.execute_task(task_id))
        
        return task_id
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task
        Returns the result of the task execution
        """
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return {"success": False, "error": "Task not found"}
        
        task = self.tasks[task_id]
        
        # Check if the task is already running or completed
        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not pending (status: {task.status.value})")
            return {"success": False, "error": f"Task is not pending (status: {task.status.value})"}
        
        # Check if the agent exists
        if task.agent_id not in self.agents:
            logger.error(f"Agent not found: {task.agent_id}")
            return {"success": False, "error": "Agent not found"}
        
        agent = self.agents[task.agent_id]
        
        # Update task and agent status
        task.status = TaskStatus.RUNNING
        task.updated_at = time.time()
        agent.status = AgentStatus.ACTIVE
        agent.current_task_id = task_id
        agent.updated_at = time.time()
        
        # Save data
        self._save_data()
        
        logger.info(f"Executing task {task_id} with agent {agent.agent_id}")
        
        try:
            # In a real implementation, this would use the agent's model to execute the task
            # For now, we'll simulate task execution
            result = await self._simulate_task_execution(task, agent)
            
            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED if result.get("success", False) else TaskStatus.FAILED
            task.progress = 100 if result.get("success", False) else task.progress
            task.updated_at = time.time()
            
            # Update agent stats
            if result.get("success", False):
                agent.tasks_completed += 1
            else:
                agent.tasks_failed += 1
            
            agent.status = AgentStatus.IDLE
            agent.current_task_id = None
            agent.updated_at = time.time()
            
            # Save data
            self._save_data()
            
            logger.info(f"Task {task_id} completed with status: {task.status.value}")
            return result
        
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            
            # Update task and agent status
            task.status = TaskStatus.FAILED
            task.result = {"success": False, "error": str(e)}
            task.updated_at = time.time()
            
            agent.status = AgentStatus.IDLE
            agent.current_task_id = None
            agent.tasks_failed += 1
            agent.updated_at = time.time()
            
            # Save data
            self._save_data()
            
            return {"success": False, "error": str(e)}
    
    async def _simulate_task_execution(self, task: Task, agent: Microagent) -> Dict[str, Any]:
        """Simulate task execution (for demonstration)"""
        # Simulate task execution time based on priority and complexity
        execution_time = 2 + (5 - task.priority) * 0.5
        
        # Update progress periodically
        for progress in range(10, 100, 10):
            await asyncio.sleep(execution_time / 10)
            task.progress = progress
            task.updated_at = time.time()
            self._save_data()
        
        # Simulate task result
        if task.execution_mode == "simulation":
            # In simulation mode, always succeed
            return {
                "success": True,
                "message": f"Task simulated successfully by {agent.config.name}",
                "execution_mode": "simulation",
                "agent_type": agent.config.agent_type.value,
                "duration_seconds": execution_time,
                "artifacts": [
                    {"type": "log", "content": f"Simulated execution of: {task.description}"},
                    {"type": "summary", "content": f"Task completed in simulation mode"}
                ]
            }
        else:
            # In live mode, 80% chance of success
            success = random.random() < 0.8
            
            if success:
                return {
                    "success": True,
                    "message": f"Task executed successfully by {agent.config.name}",
                    "execution_mode": "live",
                    "agent_type": agent.config.agent_type.value,
                    "duration_seconds": execution_time,
                    "artifacts": [
                        {"type": "log", "content": f"Live execution of: {task.description}"},
                        {"type": "summary", "content": f"Task completed in live mode"}
                    ]
                }
            else:
                return {
                    "success": False,
                    "message": f"Task execution failed",
                    "execution_mode": "live",
                    "agent_type": agent.config.agent_type.value,
                    "duration_seconds": execution_time,
                    "error": "Simulated failure in live execution mode",
                    "artifacts": [
                        {"type": "log", "content": f"Live execution of: {task.description}"},
                        {"type": "error", "content": f"Encountered error during execution"}
                    ]
                }
    
    async def process_message(self, message: str, user_id: str = "user") -> Dict[str, Any]:
        """
        Process a user message and determine if it should spawn an agent or create a task
        Returns information about the action taken
        """
        # Check if the message is task-like
        is_task = self.is_task_like(message)
        
        if is_task:
            # Identify the appropriate agent type
            agent_type, confidence = self.identify_agent_type(message)
            
            # If confidence is high enough, create a task
            if confidence >= 0.6:
                # Find or spawn an agent
                matching_agents = [a for a in self.agents.values() 
                                 if a.config.agent_type == agent_type and a.status == AgentStatus.IDLE]
                
                agent_id = None
                if matching_agents:
                    agent_id = matching_agents[0].agent_id
                else:
                    agent_id = await self.spawn_agent(agent_type)
                
                if agent_id:
                    # Create a task
                    task_id = await self.create_task(message, agent_id)
                    
                    if task_id:
                        return {
                            "action": "task_created",
                            "task_id": task_id,
                            "agent_id": agent_id,
                            "agent_type": agent_type.value,
                            "confidence": confidence,
                            "message": f"Created task and assigned to {agent_type.value} agent"
                        }
            
            # If confidence is low or task creation failed, just return the analysis
            return {
                "action": "analyzed",
                "is_task": is_task,
                "agent_type": agent_type.value,
                "confidence": confidence,
                "message": f"Message appears to be a task for a {agent_type.value} agent (confidence: {confidence:.2f})"
            }
        
        # Not a task, just return the analysis
        return {
            "action": "analyzed",
            "is_task": False,
            "message": "Message does not appear to be a task"
        }
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        agent_dict = asdict(agent)
        
        # Convert enums to strings for JSON serialization
        agent_dict["status"] = agent_dict["status"].value
        agent_dict["config"]["agent_type"] = agent_dict["config"]["agent_type"].value
        
        return agent_dict
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        task_dict = asdict(task)
        
        # Convert enum to string for JSON serialization
        task_dict["status"] = task_dict["status"].value
        
        return task_dict
    
    def list_agents(self, status: Optional[Union[AgentStatus, str]] = None,
                  agent_type: Optional[Union[AgentType, str]] = None) -> List[Dict[str, Any]]:
        """
        List agents
        Optionally filtered by status and type
        """
        # Convert string to enum if needed
        if isinstance(status, str):
            try:
                status = AgentStatus(status)
            except ValueError:
                status = None
        
        if isinstance(agent_type, str):
            try:
                agent_type = AgentType(agent_type)
            except ValueError:
                agent_type = None
        
        filtered_agents = []
        
        for agent in self.agents.values():
            if (status is None or agent.status == status) and \
               (agent_type is None or agent.config.agent_type == agent_type):
                agent_dict = asdict(agent)
                
                # Convert enums to strings for JSON serialization
                agent_dict["status"] = agent_dict["status"].value
                agent_dict["config"]["agent_type"] = agent_dict["config"]["agent_type"].value
                
                filtered_agents.append(agent_dict)
        
        # Sort by updated_at (newest first)
        filtered_agents.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return filtered_agents
    
    def list_tasks(self, status: Optional[Union[TaskStatus, str]] = None,
                 agent_id: Optional[str] = None,
                 parent_task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks
        Optionally filtered by status, agent ID, and parent task ID
        """
        # Convert string to enum if needed
        if isinstance(status, str):
            try:
                status = TaskStatus(status)
            except ValueError:
                status = None
        
        filtered_tasks = []
        
        for task in self.tasks.values():
            if (status is None or task.status == status) and \
               (agent_id is None or task.agent_id == agent_id) and \
               (parent_task_id is None or task.parent_task_id == parent_task_id):
                task_dict = asdict(task)
                
                # Convert enum to string for JSON serialization
                task_dict["status"] = task_dict["status"].value
                
                filtered_tasks.append(task_dict)
        
        # Sort by priority (highest first) and then by created_at (newest first)
        filtered_tasks.sort(key=lambda x: (-x["priority"], -x["created_at"]))
        
        return filtered_tasks
    
    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent
        Returns True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        agent = self.agents[agent_id]
        
        # Cancel any running task
        if agent.current_task_id and agent.current_task_id in self.tasks:
            task = self.tasks[agent.current_task_id]
            task.status = TaskStatus.CANCELLED
            task.updated_at = time.time()
        
        # Update agent status
        agent.status = AgentStatus.TERMINATED
        agent.updated_at = time.time()
        
        # Save data
        self._save_data()
        
        logger.info(f"Terminated agent: {agent_id}")
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        Returns True if successful, False otherwise
        """
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        # Can only cancel pending or running tasks
        if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            logger.warning(f"Cannot cancel task {task_id} with status {task.status.value}")
            return False
        
        # Update task status
        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()
        
        # If the task is running, update the agent
        if task.status == TaskStatus.RUNNING and task.agent_id in self.agents:
            agent = self.agents[task.agent_id]
            if agent.current_task_id == task_id:
                agent.status = AgentStatus.IDLE
                agent.current_task_id = None
                agent.updated_at = time.time()
        
        # Cancel any subtasks
        for subtask_id in task.subtasks:
            if subtask_id in self.tasks:
                await self.cancel_task(subtask_id)
        
        # Save data
        self._save_data()
        
        logger.info(f"Cancelled task: {task_id}")
        return True
    
    def get_agent_types(self) -> List[Dict[str, Any]]:
        """Get information about available agent types"""
        agent_types = []
        
        for agent_type, config in self.agent_configs.items():
            agent_types.append({
                "type": agent_type.value,
                "name": config.name,
                "description": config.description,
                "capabilities": config.capabilities,
                "triggers": config.triggers
            })
        
        return agent_types
    
    def cleanup_terminated_agents(self) -> int:
        """
        Remove terminated agents from memory
        Returns the number of agents removed
        """
        terminated_agents = [agent_id for agent_id, agent in self.agents.items()
                           if agent.status == AgentStatus.TERMINATED]
        
        for agent_id in terminated_agents:
            del self.agents[agent_id]
        
        # Save data
        if terminated_agents:
            self._save_data()
        
        logger.info(f"Cleaned up {len(terminated_agents)} terminated agents")
        return len(terminated_agents)
    
    def cleanup_old_tasks(self, days: int = 7) -> int:
        """
        Remove old completed, failed, or cancelled tasks
        Returns the number of tasks removed
        """
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        old_tasks = [task_id for task_id, task in self.tasks.items()
                   if task.updated_at < cutoff_time and
                   task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
        
        for task_id in old_tasks:
            del self.tasks[task_id]
        
        # Save data
        if old_tasks:
            self._save_data()
        
        logger.info(f"Cleaned up {len(old_tasks)} old tasks")
        return len(old_tasks)

# Create singleton instance
microagent_launcher = MicroagentLauncher()

# For testing
if __name__ == "__main__":
    async def test_microagent_launcher():
        # Process a task-like message
        result = await microagent_launcher.process_message("Create a Python function to calculate Fibonacci numbers")
        print("Process message result:", result)
        
        if result.get("action") == "task_created":
            task_id = result.get("task_id")
            
            # Wait for task to complete
            while True:
                task = microagent_launcher.get_task(task_id)
                if task["status"] in ["completed", "failed", "cancelled"]:
                    break
                print(f"Task progress: {task['progress']}%")
                await asyncio.sleep(1)
            
            print("Task result:", task["result"])
        
        # List agents
        agents = microagent_launcher.list_agents()
        print(f"Active agents: {len(agents)}")
        
        # List tasks
        tasks = microagent_launcher.list_tasks()
        print(f"Tasks: {len(tasks)}")
    
    asyncio.run(test_microagent_launcher())