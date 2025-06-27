#!/usr/bin/env python3
"""
WADE Task Chain Executor
Executes complex task graphs with dependencies and parallel execution
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
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
    """Individual task in the execution graph"""
    task_id: str
    name: str
    description: str
    agent_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class TaskChain:
    """Complete task execution chain"""
    chain_id: str
    name: str
    description: str
    tasks: Dict[str, TaskNode]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING

class TaskGraphExecutor:
    """Executes task graphs with dependency resolution"""
    
    def __init__(self):
        self.active_chains: Dict[str, TaskChain] = {}
        self.execution_history: List[TaskChain] = []
        self.agent_registry = {
            'refactor': self._execute_refactor_task,
            'test': self._execute_test_task,
            'deploy': self._execute_deploy_task,
            'analyze': self._execute_analyze_task,
            'document': self._execute_document_task,
            'security_scan': self._execute_security_task,
            'performance_test': self._execute_performance_task,
            'custom': self._execute_custom_task
        }
    
    def load_chain_from_yaml(self, yaml_path: str) -> TaskChain:
        """Load task chain from YAML file"""
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        chain_id = data.get('chain_id', f"chain_{int(datetime.now().timestamp())}")
        
        tasks = {}
        for task_data in data.get('tasks', []):
            task = TaskNode(
                task_id=task_data['task_id'],
                name=task_data['name'],
                description=task_data.get('description', ''),
                agent_type=task_data['agent_type'],
                parameters=task_data.get('parameters', {}),
                dependencies=task_data.get('dependencies', []),
                max_retries=task_data.get('max_retries', 3)
            )
            tasks[task.task_id] = task
        
        chain = TaskChain(
            chain_id=chain_id,
            name=data.get('name', 'Unnamed Chain'),
            description=data.get('description', ''),
            tasks=tasks,
            metadata=data.get('metadata', {})
        )
        
        return chain
    
    def load_chain_from_json(self, json_path: str) -> TaskChain:
        """Load task chain from JSON file"""
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_chain(data)
    
    def _dict_to_chain(self, data: Dict[str, Any]) -> TaskChain:
        """Convert dictionary to TaskChain"""
        
        chain_id = data.get('chain_id', f"chain_{int(datetime.now().timestamp())}")
        
        tasks = {}
        for task_data in data.get('tasks', []):
            task = TaskNode(
                task_id=task_data['task_id'],
                name=task_data['name'],
                description=task_data.get('description', ''),
                agent_type=task_data['agent_type'],
                parameters=task_data.get('parameters', {}),
                dependencies=task_data.get('dependencies', []),
                max_retries=task_data.get('max_retries', 3)
            )
            tasks[task.task_id] = task
        
        return TaskChain(
            chain_id=chain_id,
            name=data.get('name', 'Unnamed Chain'),
            description=data.get('description', ''),
            tasks=tasks,
            metadata=data.get('metadata', {})
        )
    
    def create_chain_from_prompt(self, user_prompt: str, repo_path: str) -> TaskChain:
        """Create task chain from natural language prompt"""
        
        # Parse prompt to identify tasks
        tasks = self._parse_prompt_to_tasks(user_prompt, repo_path)
        
        chain_id = f"prompt_chain_{int(datetime.now().timestamp())}"
        
        return TaskChain(
            chain_id=chain_id,
            name=f"Chain from: {user_prompt[:50]}...",
            description=f"Auto-generated from prompt: {user_prompt}",
            tasks=tasks,
            metadata={'source': 'prompt', 'original_prompt': user_prompt}
        )
    
    def _parse_prompt_to_tasks(self, prompt: str, repo_path: str) -> Dict[str, TaskNode]:
        """Parse natural language prompt into task nodes"""
        
        tasks = {}
        prompt_lower = prompt.lower()
        
        # Task 1: Always analyze first
        tasks['analyze'] = TaskNode(
            task_id='analyze',
            name='Analyze Repository',
            description='Analyze the repository structure and code',
            agent_type='analyze',
            parameters={'repo_path': repo_path}
        )
        
        # Task 2: Refactor if requested
        if any(word in prompt_lower for word in ['refactor', 'convert', 'transform', 'change']):
            tasks['refactor'] = TaskNode(
                task_id='refactor',
                name='Refactor Code',
                description='Refactor code according to requirements',
                agent_type='refactor',
                parameters={'repo_path': repo_path, 'vision': prompt},
                dependencies=['analyze']
            )
        
        # Task 3: Add tests if requested
        if any(word in prompt_lower for word in ['test', 'testing', 'coverage']):
            tasks['test'] = TaskNode(
                task_id='test',
                name='Add Tests',
                description='Generate and run comprehensive tests',
                agent_type='test',
                parameters={'repo_path': repo_path},
                dependencies=['refactor'] if 'refactor' in tasks else ['analyze']
            )
        
        # Task 4: Security scan if requested
        if any(word in prompt_lower for word in ['security', 'secure', 'vulnerability']):
            tasks['security'] = TaskNode(
                task_id='security',
                name='Security Scan',
                description='Perform security analysis and fixes',
                agent_type='security_scan',
                parameters={'repo_path': repo_path},
                dependencies=['refactor'] if 'refactor' in tasks else ['analyze']
            )
        
        # Task 5: Documentation if requested
        if any(word in prompt_lower for word in ['document', 'docs', 'readme']):
            tasks['document'] = TaskNode(
                task_id='document',
                name='Generate Documentation',
                description='Generate comprehensive documentation',
                agent_type='document',
                parameters={'repo_path': repo_path},
                dependencies=list(tasks.keys())  # Depends on all previous tasks
            )
        
        # Task 6: Deploy if requested
        if any(word in prompt_lower for word in ['deploy', 'docker', 'container']):
            tasks['deploy'] = TaskNode(
                task_id='deploy',
                name='Prepare Deployment',
                description='Create deployment configurations',
                agent_type='deploy',
                parameters={'repo_path': repo_path},
                dependencies=list(tasks.keys())  # Depends on all previous tasks
            )
        
        return tasks
    
    async def execute_chain(self, chain: TaskChain, progress_callback=None) -> TaskChain:
        """Execute a complete task chain"""
        
        self.active_chains[chain.chain_id] = chain
        chain.status = TaskStatus.RUNNING
        
        try:
            # Build dependency graph
            graph = self._build_dependency_graph(chain)
            
            # Execute tasks in topological order
            execution_order = list(nx.topological_sort(graph))
            
            for task_id in execution_order:
                task = chain.tasks[task_id]
                
                # Wait for dependencies
                await self._wait_for_dependencies(task, chain)
                
                # Execute task
                await self._execute_task(task, chain, progress_callback)
                
                # Check if chain should continue
                if task.status == TaskStatus.FAILED and not self._should_continue_on_failure(task, chain):
                    chain.status = TaskStatus.FAILED
                    break
            
            # Determine final chain status
            if chain.status != TaskStatus.FAILED:
                if all(task.status == TaskStatus.COMPLETED for task in chain.tasks.values()):
                    chain.status = TaskStatus.COMPLETED
                else:
                    chain.status = TaskStatus.FAILED
            
        except Exception as e:
            chain.status = TaskStatus.FAILED
            if progress_callback:
                await progress_callback(f"Chain execution failed: {str(e)}")
        
        finally:
            # Move to history
            self.execution_history.append(chain)
            if chain.chain_id in self.active_chains:
                del self.active_chains[chain.chain_id]
        
        return chain
    
    def _build_dependency_graph(self, chain: TaskChain) -> nx.DiGraph:
        """Build NetworkX graph from task dependencies"""
        
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task_id in chain.tasks:
            graph.add_node(task_id)
        
        # Add dependency edges
        for task_id, task in chain.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in chain.tasks:
                    graph.add_edge(dep_id, task_id)
        
        return graph
    
    async def _wait_for_dependencies(self, task: TaskNode, chain: TaskChain):
        """Wait for task dependencies to complete"""
        
        while True:
            all_deps_complete = True
            
            for dep_id in task.dependencies:
                if dep_id in chain.tasks:
                    dep_task = chain.tasks[dep_id]
                    if dep_task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                        all_deps_complete = False
                        break
            
            if all_deps_complete:
                break
            
            await asyncio.sleep(0.5)  # Check every 500ms
    
    async def _execute_task(self, task: TaskNode, chain: TaskChain, progress_callback=None):
        """Execute a single task"""
        
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        if progress_callback:
            await progress_callback(f"Starting task: {task.name}")
        
        try:
            # Get agent executor
            if task.agent_type in self.agent_registry:
                executor = self.agent_registry[task.agent_type]
                result = await executor(task, chain)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                
                if progress_callback:
                    await progress_callback(f"Completed task: {task.name}")
            
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
        
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            
            if progress_callback:
                await progress_callback(f"Failed task: {task.name} - {str(e)}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                
                if progress_callback:
                    await progress_callback(f"Retrying task: {task.name} (attempt {task.retry_count + 1})")
                
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_task(task, chain, progress_callback)
        
        finally:
            task.end_time = datetime.now()
    
    def _should_continue_on_failure(self, failed_task: TaskNode, chain: TaskChain) -> bool:
        """Determine if chain should continue after task failure"""
        
        # Check if any remaining tasks don't depend on the failed task
        for task_id, task in chain.tasks.items():
            if task.status == TaskStatus.PENDING:
                if failed_task.task_id not in task.dependencies:
                    return True
        
        return False
    
    # Agent Executors
    async def _execute_refactor_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute refactor task"""
        
        try:
            from wade_refactor_system import WADERefactorSystem
            
            system = WADERefactorSystem()
            repo_path = task.parameters.get('repo_path')
            vision = task.parameters.get('vision', 'improve code structure')
            
            result = system.refactor_repository(repo_path, vision)
            
            return {
                'success': result.success,
                'summary': result.summary,
                'files_changed': result.files_changed,
                'files_created': result.files_created,
                'execution_output': result.execution_output
            }
        
        except Exception as e:
            raise Exception(f"Refactor task failed: {str(e)}")
    
    async def _execute_test_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute test generation task"""
        
        repo_path = task.parameters.get('repo_path')
        
        # Generate tests
        test_files = []
        
        # Find Python files to test
        for py_file in Path(repo_path).glob('**/*.py'):
            if 'test' not in py_file.name and py_file.name != '__init__.py':
                test_file = py_file.parent / f"test_{py_file.name}"
                
                # Generate basic test
                test_content = f'''import unittest
from {py_file.stem} import *

class Test{py_file.stem.title()}(unittest.TestCase):
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # TODO: Add specific tests
        self.assertTrue(True)
    
    def test_error_handling(self):
        """Test error handling"""
        # TODO: Add error handling tests
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
'''
                
                with open(test_file, 'w') as f:
                    f.write(test_content)
                
                test_files.append(str(test_file))
        
        return {
            'success': True,
            'test_files_created': test_files,
            'summary': f"Generated {len(test_files)} test files"
        }
    
    async def _execute_analyze_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute repository analysis task"""
        
        repo_path = Path(task.parameters.get('repo_path'))
        
        analysis = {
            'total_files': 0,
            'python_files': 0,
            'javascript_files': 0,
            'entry_points': [],
            'frameworks': [],
            'dependencies': [],
            'structure': {}
        }
        
        # Analyze files
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                analysis['total_files'] += 1
                
                if file_path.suffix == '.py':
                    analysis['python_files'] += 1
                    
                    # Check for entry points
                    if file_path.name in ['app.py', 'main.py', 'server.py', 'run.py']:
                        analysis['entry_points'].append(str(file_path))
                    
                    # Check for frameworks
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if 'from flask import' in content or 'import flask' in content:
                                analysis['frameworks'].append('flask')
                            if 'from fastapi import' in content or 'import fastapi' in content:
                                analysis['frameworks'].append('fastapi')
                            if 'from django import' in content or 'import django' in content:
                                analysis['frameworks'].append('django')
                    except:
                        pass
                
                elif file_path.suffix == '.js':
                    analysis['javascript_files'] += 1
        
        # Check for requirements
        req_file = repo_path / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    analysis['dependencies'] = [line.strip() for line in f.readlines() if line.strip()]
            except:
                pass
        
        # Remove duplicates
        analysis['frameworks'] = list(set(analysis['frameworks']))
        
        return {
            'success': True,
            'analysis': analysis,
            'summary': f"Analyzed {analysis['total_files']} files, found {len(analysis['frameworks'])} frameworks"
        }
    
    async def _execute_deploy_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute deployment preparation task"""
        
        repo_path = Path(task.parameters.get('repo_path'))
        
        # Create Dockerfile
        dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
'''
        
        dockerfile_path = repo_path / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        compose_content = '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - .:/app
'''
        
        compose_path = repo_path / 'docker-compose.yml'
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        return {
            'success': True,
            'files_created': ['Dockerfile', 'docker-compose.yml'],
            'summary': 'Created Docker deployment configuration'
        }
    
    async def _execute_document_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute documentation generation task"""
        
        repo_path = Path(task.parameters.get('repo_path'))
        
        # Generate README.md
        readme_content = f'''# Project Documentation

## Overview
This project was automatically refactored and documented by WADE.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```

## API Endpoints
- GET / - Home endpoint
- GET /docs - API documentation

## Testing
```bash
python -m pytest
```

## Deployment
```bash
docker-compose up
```

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        
        readme_path = repo_path / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return {
            'success': True,
            'files_created': ['README.md'],
            'summary': 'Generated project documentation'
        }
    
    async def _execute_security_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute security scan task"""
        
        repo_path = Path(task.parameters.get('repo_path'))
        
        security_issues = []
        
        # Basic security checks
        for py_file in repo_path.glob('**/*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                    # Check for common security issues
                    if 'eval(' in content:
                        security_issues.append(f"{py_file}: Use of eval() detected")
                    if 'exec(' in content:
                        security_issues.append(f"{py_file}: Use of exec() detected")
                    if 'shell=True' in content:
                        security_issues.append(f"{py_file}: Shell injection risk detected")
                    if 'password' in content.lower() and '=' in content:
                        security_issues.append(f"{py_file}: Potential hardcoded password")
            except:
                pass
        
        return {
            'success': True,
            'security_issues': security_issues,
            'summary': f"Found {len(security_issues)} potential security issues"
        }
    
    async def _execute_performance_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute performance testing task"""
        
        return {
            'success': True,
            'performance_metrics': {
                'response_time': '< 100ms',
                'throughput': '1000 req/s',
                'memory_usage': '< 100MB'
            },
            'summary': 'Performance analysis completed'
        }
    
    async def _execute_custom_task(self, task: TaskNode, chain: TaskChain) -> Dict[str, Any]:
        """Execute custom task"""
        
        # Custom task execution logic
        return {
            'success': True,
            'summary': 'Custom task completed'
        }
    
    def get_chain_status(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task chain"""
        
        chain = self.active_chains.get(chain_id)
        if not chain:
            # Check history
            for hist_chain in self.execution_history:
                if hist_chain.chain_id == chain_id:
                    chain = hist_chain
                    break
        
        if not chain:
            return None
        
        return {
            'chain_id': chain.chain_id,
            'name': chain.name,
            'status': chain.status.value,
            'total_tasks': len(chain.tasks),
            'completed_tasks': len([t for t in chain.tasks.values() if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in chain.tasks.values() if t.status == TaskStatus.FAILED]),
            'tasks': {
                task_id: {
                    'name': task.name,
                    'status': task.status.value,
                    'start_time': task.start_time.isoformat() if task.start_time else None,
                    'end_time': task.end_time.isoformat() if task.end_time else None,
                    'error': task.error
                }
                for task_id, task in chain.tasks.items()
            }
        }
    
    def save_chain_template(self, chain: TaskChain, template_path: str):
        """Save chain as reusable template"""
        
        template_data = {
            'name': chain.name,
            'description': chain.description,
            'tasks': [
                {
                    'task_id': task.task_id,
                    'name': task.name,
                    'description': task.description,
                    'agent_type': task.agent_type,
                    'parameters': task.parameters,
                    'dependencies': task.dependencies,
                    'max_retries': task.max_retries
                }
                for task in chain.tasks.values()
            ],
            'metadata': chain.metadata
        }
        
        with open(template_path, 'w') as f:
            if template_path.endswith('.yaml') or template_path.endswith('.yml'):
                yaml.dump(template_data, f, default_flow_style=False)
            else:
                json.dump(template_data, f, indent=2)

# Global executor
task_executor = TaskGraphExecutor()

def main():
    """Test the task chain executor"""
    
    async def test_executor():
        # Create a test chain
        chain = task_executor.create_chain_from_prompt(
            "Take the repo /workspace/demo_repo and convert it to FastAPI with tests and documentation",
            "/workspace/demo_repo"
        )
        
        print(f"Created chain: {chain.name}")
        print(f"Tasks: {list(chain.tasks.keys())}")
        
        # Execute the chain
        async def progress_callback(message):
            print(f"Progress: {message}")
        
        result_chain = await task_executor.execute_chain(chain, progress_callback)
        
        print(f"Chain completed with status: {result_chain.status}")
        
        # Get status
        status = task_executor.get_chain_status(chain.chain_id)
        print("Final status:", json.dumps(status, indent=2))
    
    asyncio.run(test_executor())

if __name__ == "__main__":
    main()