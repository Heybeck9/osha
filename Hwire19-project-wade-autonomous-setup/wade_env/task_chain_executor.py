#!/usr/bin/env python3
"""
WADE Task Chain Executor - DAG-based autonomous task execution
Handles complex multi-step operations with dependencies
"""

import json
import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskNode:
    task_id: str
    name: str
    agent: str
    depends_on: List[str]
    output_var: str
    input_vars: List[str]
    execution_mode: str = "simulation"
    priority: int = 1
    timeout: int = 300
    retry_count: int = 3
    status: TaskStatus = TaskStatus.PENDING
    output: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_log: List[str] = None

    def __post_init__(self):
        if self.execution_log is None:
            self.execution_log = []

@dataclass
class TaskChain:
    chain_id: str
    name: str
    description: str
    tasks: List[TaskNode]
    global_vars: Dict[str, Any]
    execution_mode: str = "simulation"
    created_at: datetime = None
    status: TaskStatus = TaskStatus.PENDING
    success_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class TaskChainExecutor:
    def __init__(self, workspace_path: str = "/workspace/wade_env"):
        self.workspace_path = workspace_path
        self.active_chains: Dict[str, TaskChain] = {}
        self.execution_graph = nx.DiGraph()
        self.global_context: Dict[str, Any] = {}
        
    def create_chain_from_json(self, chain_config: Dict) -> TaskChain:
        """Create a task chain from JSON configuration"""
        chain_id = chain_config.get("chain_id", str(uuid.uuid4()))
        
        tasks = []
        for task_config in chain_config.get("tasks", []):
            task = TaskNode(
                task_id=task_config.get("task_id", str(uuid.uuid4())),
                name=task_config["name"],
                agent=task_config["agent"],
                depends_on=task_config.get("depends_on", []),
                output_var=task_config.get("output_var", f"output_{len(tasks)}"),
                input_vars=task_config.get("input_vars", []),
                execution_mode=task_config.get("execution_mode", "simulation"),
                priority=task_config.get("priority", 1),
                timeout=task_config.get("timeout", 300),
                retry_count=task_config.get("retry_count", 3)
            )
            tasks.append(task)
        
        chain = TaskChain(
            chain_id=chain_id,
            name=chain_config["name"],
            description=chain_config.get("description", ""),
            tasks=tasks,
            global_vars=chain_config.get("global_vars", {}),
            execution_mode=chain_config.get("execution_mode", "simulation")
        )
        
        self.active_chains[chain_id] = chain
        self._build_execution_graph(chain)
        return chain
    
    def _build_execution_graph(self, chain: TaskChain):
        """Build NetworkX graph for dependency resolution"""
        graph_id = f"chain_{chain.chain_id}"
        
        # Add nodes
        for task in chain.tasks:
            self.execution_graph.add_node(
                task.task_id,
                task=task,
                chain_id=chain.chain_id
            )
        
        # Add edges for dependencies
        for task in chain.tasks:
            for dependency in task.depends_on:
                self.execution_graph.add_edge(dependency, task.task_id)
    
    async def execute_chain(self, chain_id: str) -> TaskChain:
        """Execute a complete task chain with dependency resolution"""
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.active_chains[chain_id]
        chain.status = TaskStatus.RUNNING
        
        try:
            # Get topological order for execution
            execution_order = list(nx.topological_sort(
                self.execution_graph.subgraph([t.task_id for t in chain.tasks])
            ))
            
            print(f"🔗 Executing chain '{chain.name}' with {len(chain.tasks)} tasks")
            print(f"📋 Execution order: {' → '.join([self._get_task_name(tid) for tid in execution_order])}")
            
            # Execute tasks in dependency order
            for task_id in execution_order:
                task = self._get_task_by_id(chain, task_id)
                if task:
                    await self._execute_task(task, chain)
                    
                    # Stop chain if critical task fails
                    if task.status == TaskStatus.FAILED and task.priority >= 5:
                        chain.status = TaskStatus.FAILED
                        break
            
            # Calculate success score
            completed_tasks = sum(1 for t in chain.tasks if t.status == TaskStatus.COMPLETED)
            chain.success_score = completed_tasks / len(chain.tasks)
            
            if chain.success_score >= 0.8:
                chain.status = TaskStatus.COMPLETED
            else:
                chain.status = TaskStatus.FAILED
                
            print(f"✅ Chain '{chain.name}' completed with {chain.success_score:.1%} success rate")
            
        except Exception as e:
            chain.status = TaskStatus.FAILED
            print(f"❌ Chain execution failed: {str(e)}")
        
        return chain
    
    async def _execute_task(self, task: TaskNode, chain: TaskChain):
        """Execute a single task with retry logic"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        print(f"🚀 Executing task: {task.name} ({task.agent})")
        task.execution_log.append(f"[{task.start_time.strftime('%H:%M:%S')}] Starting task execution")
        
        for attempt in range(task.retry_count):
            try:
                # Prepare input context
                input_context = self._prepare_input_context(task, chain)
                
                # Execute based on agent type
                result = await self._execute_agent(task, input_context)
                
                # Store output
                task.output = result
                chain.global_vars[task.output_var] = result
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                
                duration = (task.end_time - task.start_time).total_seconds()
                task.execution_log.append(f"[{task.end_time.strftime('%H:%M:%S')}] Task completed in {duration:.1f}s")
                print(f"✅ Task '{task.name}' completed successfully")
                return
                
            except Exception as e:
                task.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Attempt {attempt + 1} failed: {str(e)}")
                if attempt == task.retry_count - 1:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.end_time = datetime.now()
                    print(f"❌ Task '{task.name}' failed after {task.retry_count} attempts")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _prepare_input_context(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Prepare input context for task execution"""
        context = {
            "task": asdict(task),
            "chain": {
                "id": chain.chain_id,
                "name": chain.name,
                "execution_mode": chain.execution_mode
            },
            "global_vars": chain.global_vars.copy(),
            "workspace": self.workspace_path
        }
        
        # Add specific input variables
        for var_name in task.input_vars:
            if var_name in chain.global_vars:
                context[var_name] = chain.global_vars[var_name]
        
        return context
    
    async def _execute_agent(self, task: TaskNode, context: Dict[str, Any]) -> Any:
        """Execute specific agent based on task configuration"""
        agent_name = task.agent.lower()
        
        # Route to appropriate agent
        if agent_name == "planner":
            return await self._execute_planner_agent(task, context)
        elif agent_name == "coder":
            return await self._execute_coder_agent(task, context)
        elif agent_name == "tester":
            return await self._execute_tester_agent(task, context)
        elif agent_name == "recon":
            return await self._execute_recon_agent(task, context)
        elif agent_name == "payload":
            return await self._execute_payload_agent(task, context)
        else:
            return await self._execute_generic_agent(task, context)
    
    async def _execute_planner_agent(self, task: TaskNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning agent"""
        await asyncio.sleep(2)  # Simulate planning time
        
        plan = {
            "strategy": f"Multi-step approach for {task.name}",
            "steps": [
                "Analyze requirements",
                "Design architecture", 
                "Implement solution",
                "Test and validate"
            ],
            "estimated_time": "15-30 minutes",
            "risk_level": "low" if context["chain"]["execution_mode"] == "simulation" else "medium"
        }
        
        return plan
    
    async def _execute_coder_agent(self, task: TaskNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coding agent"""
        await asyncio.sleep(3)  # Simulate coding time
        
        code_result = {
            "files_created": [
                f"{context['workspace']}/generated_code.py",
                f"{context['workspace']}/requirements.txt"
            ],
            "functions": ["main()", "process_data()", "validate_input()"],
            "lines_of_code": 150,
            "test_coverage": "85%"
        }
        
        return code_result
    
    async def _execute_tester_agent(self, task: TaskNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing agent"""
        await asyncio.sleep(2)  # Simulate testing time
        
        test_result = {
            "tests_run": 12,
            "tests_passed": 11,
            "tests_failed": 1,
            "coverage": "92%",
            "performance": "Good",
            "security_issues": 0 if context["chain"]["execution_mode"] == "simulation" else 1
        }
        
        return test_result
    
    async def _execute_recon_agent(self, task: TaskNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reconnaissance agent"""
        await asyncio.sleep(4)  # Simulate recon time
        
        if context["chain"]["execution_mode"] == "simulation":
            recon_result = {
                "mode": "simulation",
                "targets_found": ["example.com", "test.local"],
                "ports_scanned": [80, 443, 22],
                "vulnerabilities": ["Simulated XSS", "Simulated SQLi"],
                "risk_assessment": "Educational purposes only"
            }
        else:
            recon_result = {
                "mode": "live",
                "warning": "Live reconnaissance requires explicit authorization",
                "targets_found": [],
                "ports_scanned": [],
                "vulnerabilities": [],
                "risk_assessment": "High - requires user approval"
            }
        
        return recon_result
    
    async def _execute_payload_agent(self, task: TaskNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute payload generation agent"""
        await asyncio.sleep(3)  # Simulate payload generation
        
        if context["chain"]["execution_mode"] == "simulation":
            payload_result = {
                "mode": "simulation",
                "payload_type": "Educational demonstration",
                "file_path": f"{context['workspace']}/payloads/demo_payload.py",
                "hash": hashlib.sha256(b"demo_payload_content").hexdigest()[:16],
                "safety": "Sandboxed - no real impact"
            }
        else:
            payload_result = {
                "mode": "live",
                "warning": "Live payload generation requires explicit approval",
                "payload_type": "Requires user confirmation",
                "file_path": None,
                "hash": None,
                "safety": "User responsibility"
            }
        
        return payload_result
    
    async def _execute_generic_agent(self, task: TaskNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic agent"""
        await asyncio.sleep(1)  # Simulate generic work
        
        return {
            "agent": task.agent,
            "task": task.name,
            "status": "completed",
            "execution_mode": context["chain"]["execution_mode"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_task_by_id(self, chain: TaskChain, task_id: str) -> Optional[TaskNode]:
        """Get task by ID from chain"""
        for task in chain.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def _get_task_name(self, task_id: str) -> str:
        """Get task name by ID"""
        for chain in self.active_chains.values():
            for task in chain.tasks:
                if task.task_id == task_id:
                    return task.name
        return task_id
    
    def get_chain_status(self, chain_id: str) -> Dict[str, Any]:
        """Get detailed chain execution status"""
        if chain_id not in self.active_chains:
            return {"error": "Chain not found"}
        
        chain = self.active_chains[chain_id]
        
        return {
            "chain_id": chain.chain_id,
            "name": chain.name,
            "status": chain.status.value,
            "success_score": chain.success_score,
            "execution_mode": chain.execution_mode,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "agent": task.agent,
                    "status": task.status.value,
                    "progress": 100 if task.status == TaskStatus.COMPLETED else 
                              50 if task.status == TaskStatus.RUNNING else 0,
                    "execution_log": task.execution_log[-5:],  # Last 5 log entries
                    "output_summary": str(task.output)[:200] if task.output else None
                }
                for task in chain.tasks
            ]
        }

# Example usage and test chains
EXAMPLE_CHAINS = {
    "web_scraper_chain": {
        "name": "Web Scraper Development",
        "description": "Complete web scraper development and testing pipeline",
        "execution_mode": "simulation",
        "global_vars": {
            "target_url": "https://example.com",
            "output_format": "json"
        },
        "tasks": [
            {
                "name": "Plan scraper architecture",
                "agent": "Planner",
                "depends_on": [],
                "output_var": "architecture_plan",
                "input_vars": ["target_url"]
            },
            {
                "name": "Generate scraper code",
                "agent": "Coder", 
                "depends_on": ["Plan scraper architecture"],
                "output_var": "scraper_code",
                "input_vars": ["architecture_plan", "target_url"]
            },
            {
                "name": "Test scraper functionality",
                "agent": "Tester",
                "depends_on": ["Generate scraper code"],
                "output_var": "test_results",
                "input_vars": ["scraper_code"]
            }
        ]
    },
    
    "security_assessment_chain": {
        "name": "Security Assessment Pipeline",
        "description": "Comprehensive security assessment and reporting",
        "execution_mode": "simulation",
        "global_vars": {
            "target_domain": "example.com",
            "assessment_type": "basic"
        },
        "tasks": [
            {
                "name": "Reconnaissance scan",
                "agent": "Recon",
                "depends_on": [],
                "output_var": "recon_data",
                "input_vars": ["target_domain"],
                "priority": 3
            },
            {
                "name": "Vulnerability analysis",
                "agent": "Tester",
                "depends_on": ["Reconnaissance scan"],
                "output_var": "vulnerabilities",
                "input_vars": ["recon_data"],
                "priority": 4
            },
            {
                "name": "Generate security report",
                "agent": "Coder",
                "depends_on": ["Vulnerability analysis"],
                "output_var": "security_report",
                "input_vars": ["vulnerabilities", "recon_data"],
                "priority": 2
            }
        ]
    }
}

if __name__ == "__main__":
    async def test_chain_executor():
        executor = TaskChainExecutor()
        
        # Create and execute web scraper chain
        chain = executor.create_chain_from_json(EXAMPLE_CHAINS["web_scraper_chain"])
        result = await executor.execute_chain(chain.chain_id)
        
        print("\n" + "="*50)
        print("CHAIN EXECUTION RESULTS")
        print("="*50)
        print(json.dumps(executor.get_chain_status(chain.chain_id), indent=2))
    
    asyncio.run(test_chain_executor())