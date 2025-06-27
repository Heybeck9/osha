#!/usr/bin/env python3
"""
WADE Self-Evolution Engine
Implements feedback loops, learning, prompt injection, and autonomous evolution
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import asyncio

@dataclass
class TaskExecution:
    """Record of a task execution for learning"""
    task_id: str
    user_prompt: str
    vision: str
    repo_path: str
    strategy_used: str
    success: bool
    execution_time: float
    files_changed: int
    test_results: Dict[str, Any]
    error_messages: List[str]
    user_feedback: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class EvolutionStrategy:
    """A strategy that WADE can learn and improve"""
    strategy_id: str
    name: str
    description: str
    prompt_template: str
    success_rate: float
    avg_execution_time: float
    use_count: int
    last_updated: str
    conditions: Dict[str, Any]  # When to use this strategy

class WADEEvolutionEngine:
    """Manages WADE's self-learning and evolution"""
    
    def __init__(self, db_path: str = "/workspace/wade_memory.db"):
        self.db_path = db_path
        self.strategies = {}
        self.feedback_loop = []
        self.learning_threshold = 0.7  # Success rate threshold for strategy promotion
        self._init_database()
        self._load_strategies()
    
    def _init_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Task executions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_executions (
                task_id TEXT PRIMARY KEY,
                user_prompt TEXT,
                vision TEXT,
                repo_path TEXT,
                strategy_used TEXT,
                success BOOLEAN,
                execution_time REAL,
                files_changed INTEGER,
                test_results TEXT,
                error_messages TEXT,
                user_feedback TEXT,
                timestamp TEXT
            )
        ''')
        
        # Evolution strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evolution_strategies (
                strategy_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                prompt_template TEXT,
                success_rate REAL,
                avg_execution_time REAL,
                use_count INTEGER,
                last_updated TEXT,
                conditions TEXT
            )
        ''')
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                task_id TEXT,
                rating INTEGER,
                comments TEXT,
                suggestions TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_strategies(self):
        """Load existing strategies from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM evolution_strategies')
        rows = cursor.fetchall()
        
        for row in rows:
            strategy = EvolutionStrategy(
                strategy_id=row[0],
                name=row[1],
                description=row[2],
                prompt_template=row[3],
                success_rate=row[4],
                avg_execution_time=row[5],
                use_count=row[6],
                last_updated=row[7],
                conditions=json.loads(row[8])
            )
            self.strategies[strategy.strategy_id] = strategy
        
        conn.close()
        
        # Load default strategies if none exist
        if not self.strategies:
            self._create_default_strategies()
    
    def _create_default_strategies(self):
        """Create default evolution strategies"""
        default_strategies = [
            EvolutionStrategy(
                strategy_id="flask_to_fastapi",
                name="Flask to FastAPI Conversion",
                description="Convert Flask applications to FastAPI with async endpoints",
                prompt_template="Convert {framework} to FastAPI with async endpoints and proper Pydantic models",
                success_rate=0.85,
                avg_execution_time=45.0,
                use_count=0,
                last_updated=datetime.now().isoformat(),
                conditions={"frameworks": ["flask"], "keywords": ["fastapi", "async"]}
            ),
            EvolutionStrategy(
                strategy_id="microservice_split",
                name="Microservice Architecture",
                description="Split monolithic applications into microservices",
                prompt_template="Refactor {app_type} into microservice architecture with {services} services",
                success_rate=0.72,
                avg_execution_time=120.0,
                use_count=0,
                last_updated=datetime.now().isoformat(),
                conditions={"keywords": ["microservice", "split", "services"], "file_count": ">10"}
            ),
            EvolutionStrategy(
                strategy_id="add_testing",
                name="Comprehensive Testing",
                description="Add comprehensive test suites to applications",
                prompt_template="Add comprehensive testing with {test_framework} and achieve {coverage}% coverage",
                success_rate=0.90,
                avg_execution_time=30.0,
                use_count=0,
                last_updated=datetime.now().isoformat(),
                conditions={"keywords": ["test", "testing", "coverage"], "has_tests": False}
            )
        ]
        
        for strategy in default_strategies:
            self.strategies[strategy.strategy_id] = strategy
            self._save_strategy(strategy)
    
    def _save_strategy(self, strategy: EvolutionStrategy):
        """Save strategy to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO evolution_strategies 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.strategy_id,
            strategy.name,
            strategy.description,
            strategy.prompt_template,
            strategy.success_rate,
            strategy.avg_execution_time,
            strategy.use_count,
            strategy.last_updated,
            json.dumps(strategy.conditions)
        ))
        
        conn.commit()
        conn.close()
    
    def record_execution(self, execution: TaskExecution):
        """Record a task execution for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO task_executions 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.task_id,
            execution.user_prompt,
            execution.vision,
            execution.repo_path,
            execution.strategy_used,
            execution.success,
            execution.execution_time,
            execution.files_changed,
            json.dumps(execution.test_results),
            json.dumps(execution.error_messages),
            execution.user_feedback,
            execution.timestamp
        ))
        
        conn.commit()
        conn.close()
        
        # Update strategy performance
        self._update_strategy_performance(execution)
    
    def _update_strategy_performance(self, execution: TaskExecution):
        """Update strategy performance based on execution results"""
        if execution.strategy_used in self.strategies:
            strategy = self.strategies[execution.strategy_used]
            
            # Update success rate (weighted average)
            old_weight = strategy.use_count
            new_weight = old_weight + 1
            
            strategy.success_rate = (
                (strategy.success_rate * old_weight + (1.0 if execution.success else 0.0)) / new_weight
            )
            
            # Update average execution time
            strategy.avg_execution_time = (
                (strategy.avg_execution_time * old_weight + execution.execution_time) / new_weight
            )
            
            strategy.use_count = new_weight
            strategy.last_updated = datetime.now().isoformat()
            
            self._save_strategy(strategy)
    
    def select_best_strategy(self, user_prompt: str, repo_analysis: Dict[str, Any]) -> Optional[EvolutionStrategy]:
        """Select the best strategy based on context and learning"""
        
        # Score strategies based on conditions and performance
        scored_strategies = []
        
        for strategy in self.strategies.values():
            score = self._score_strategy(strategy, user_prompt, repo_analysis)
            if score > 0:
                scored_strategies.append((score, strategy))
        
        # Sort by score and return best
        if scored_strategies:
            scored_strategies.sort(key=lambda x: x[0], reverse=True)
            return scored_strategies[0][1]
        
        return None
    
    def _score_strategy(self, strategy: EvolutionStrategy, user_prompt: str, repo_analysis: Dict[str, Any]) -> float:
        """Score a strategy based on how well it matches the current context"""
        score = 0.0
        
        # Base score from success rate
        score += strategy.success_rate * 0.4
        
        # Keyword matching
        prompt_lower = user_prompt.lower()
        if 'keywords' in strategy.conditions:
            keyword_matches = sum(1 for keyword in strategy.conditions['keywords'] 
                                if keyword in prompt_lower)
            score += (keyword_matches / len(strategy.conditions['keywords'])) * 0.3
        
        # Framework matching
        if 'frameworks' in strategy.conditions:
            detected_frameworks = repo_analysis.get('frameworks', [])
            framework_matches = sum(1 for fw in strategy.conditions['frameworks'] 
                                  if fw in detected_frameworks)
            if framework_matches > 0:
                score += 0.2
        
        # File count conditions
        if 'file_count' in strategy.conditions:
            file_count = len(repo_analysis.get('entry_points', []))
            condition = strategy.conditions['file_count']
            if condition.startswith('>') and file_count > int(condition[1:]):
                score += 0.1
            elif condition.startswith('<') and file_count < int(condition[1:]):
                score += 0.1
        
        # Penalize for low use count (exploration vs exploitation)
        if strategy.use_count < 3:
            score += 0.05  # Slight bonus for exploration
        
        return score
    
    def evolve_strategy(self, failed_execution: TaskExecution) -> EvolutionStrategy:
        """Create or evolve a strategy based on failed execution"""
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failure(failed_execution)
        
        # Create new strategy or modify existing one
        if failure_analysis['create_new']:
            return self._create_new_strategy(failed_execution, failure_analysis)
        else:
            return self._modify_existing_strategy(failed_execution, failure_analysis)
    
    def _analyze_failure(self, execution: TaskExecution) -> Dict[str, Any]:
        """Analyze why an execution failed"""
        analysis = {
            'create_new': False,
            'error_patterns': [],
            'suggested_improvements': []
        }
        
        # Analyze error messages
        for error in execution.error_messages:
            if 'syntax' in error.lower():
                analysis['error_patterns'].append('syntax_error')
                analysis['suggested_improvements'].append('Add syntax validation step')
            elif 'import' in error.lower() or 'module' in error.lower():
                analysis['error_patterns'].append('dependency_error')
                analysis['suggested_improvements'].append('Check dependencies before conversion')
            elif 'test' in error.lower():
                analysis['error_patterns'].append('test_failure')
                analysis['suggested_improvements'].append('Generate more comprehensive tests')
        
        # Check if this is a novel failure pattern
        similar_failures = self._find_similar_failures(execution)
        if len(similar_failures) < 2:
            analysis['create_new'] = True
        
        return analysis
    
    def _find_similar_failures(self, execution: TaskExecution) -> List[TaskExecution]:
        """Find similar failed executions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM task_executions 
            WHERE success = 0 AND vision LIKE ? 
            ORDER BY timestamp DESC LIMIT 10
        ''', (f'%{execution.vision}%',))
        
        rows = cursor.fetchall()
        conn.close()
        
        similar = []
        for row in rows:
            similar.append(TaskExecution(
                task_id=row[0],
                user_prompt=row[1],
                vision=row[2],
                repo_path=row[3],
                strategy_used=row[4],
                success=row[5],
                execution_time=row[6],
                files_changed=row[7],
                test_results=json.loads(row[8]),
                error_messages=json.loads(row[9]),
                user_feedback=row[10],
                timestamp=row[11]
            ))
        
        return similar
    
    def _create_new_strategy(self, execution: TaskExecution, analysis: Dict[str, Any]) -> EvolutionStrategy:
        """Create a new strategy based on failed execution"""
        
        strategy_id = f"evolved_{hashlib.md5(execution.vision.encode()).hexdigest()[:8]}"
        
        # Generate improved prompt template
        improvements = " and ".join(analysis['suggested_improvements'])
        prompt_template = f"{execution.vision} with {improvements}"
        
        new_strategy = EvolutionStrategy(
            strategy_id=strategy_id,
            name=f"Evolved: {execution.vision[:30]}...",
            description=f"Strategy evolved from failed execution: {execution.task_id}",
            prompt_template=prompt_template,
            success_rate=0.5,  # Start with neutral expectation
            avg_execution_time=execution.execution_time * 1.2,  # Expect slightly longer
            use_count=0,
            last_updated=datetime.now().isoformat(),
            conditions=self._extract_conditions_from_execution(execution)
        )
        
        self.strategies[strategy_id] = new_strategy
        self._save_strategy(new_strategy)
        
        return new_strategy
    
    def _extract_conditions_from_execution(self, execution: TaskExecution) -> Dict[str, Any]:
        """Extract conditions from execution context"""
        conditions = {}
        
        # Extract keywords from vision
        vision_words = execution.vision.lower().split()
        conditions['keywords'] = [word for word in vision_words if len(word) > 3]
        
        # Add other contextual conditions
        if 'flask' in execution.user_prompt.lower():
            conditions['frameworks'] = ['flask']
        if 'fastapi' in execution.user_prompt.lower():
            conditions['target_framework'] = 'fastapi'
        
        return conditions
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning and evolution statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total executions
        cursor.execute('SELECT COUNT(*) FROM task_executions')
        total_executions = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute('SELECT COUNT(*) FROM task_executions WHERE success = 1')
        successful_executions = cursor.fetchone()[0]
        
        # Recent performance (last 10 executions)
        cursor.execute('''
            SELECT success FROM task_executions 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        recent_results = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_executions': total_executions,
            'overall_success_rate': successful_executions / max(total_executions, 1),
            'recent_success_rate': sum(recent_results) / max(len(recent_results), 1),
            'total_strategies': len(self.strategies),
            'best_strategy': max(self.strategies.values(), key=lambda s: s.success_rate) if self.strategies else None,
            'learning_trend': 'improving' if len(recent_results) > 5 and sum(recent_results[-5:]) > sum(recent_results[:5]) else 'stable'
        }
    
    def get_strategy_recommendations(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Get strategy recommendations for a user prompt"""
        recommendations = []
        
        for strategy in self.strategies.values():
            # Simple keyword matching for recommendations
            prompt_lower = user_prompt.lower()
            relevance = 0
            
            if 'keywords' in strategy.conditions:
                matches = sum(1 for keyword in strategy.conditions['keywords'] 
                            if keyword in prompt_lower)
                relevance = matches / len(strategy.conditions['keywords'])
            
            if relevance > 0.3:  # Threshold for relevance
                recommendations.append({
                    'strategy': strategy.name,
                    'description': strategy.description,
                    'success_rate': strategy.success_rate,
                    'relevance': relevance,
                    'estimated_time': strategy.avg_execution_time
                })
        
        # Sort by relevance and success rate
        recommendations.sort(key=lambda x: (x['relevance'], x['success_rate']), reverse=True)
        return recommendations[:3]  # Top 3 recommendations

class WADETeachingInterface:
    """Interface for users to teach WADE new strategies"""
    
    def __init__(self, evolution_engine: WADEEvolutionEngine):
        self.evolution_engine = evolution_engine
    
    def create_custom_strategy(self, name: str, description: str, 
                             prompt_template: str, conditions: Dict[str, Any]) -> str:
        """Allow users to create custom strategies"""
        
        strategy_id = f"custom_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        strategy = EvolutionStrategy(
            strategy_id=strategy_id,
            name=name,
            description=description,
            prompt_template=prompt_template,
            success_rate=0.5,  # Start neutral
            avg_execution_time=60.0,  # Default estimate
            use_count=0,
            last_updated=datetime.now().isoformat(),
            conditions=conditions
        )
        
        self.evolution_engine.strategies[strategy_id] = strategy
        self.evolution_engine._save_strategy(strategy)
        
        return strategy_id
    
    def provide_feedback(self, task_id: str, rating: int, comments: str, suggestions: str):
        """Allow users to provide feedback on task executions"""
        
        feedback_id = f"feedback_{int(time.time())}"
        
        conn = sqlite3.connect(self.evolution_engine.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            task_id,
            rating,
            comments,
            suggestions,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Update the task execution with feedback
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE task_executions 
            SET user_feedback = ? 
            WHERE task_id = ?
        ''', (f"Rating: {rating}/5, Comments: {comments}", task_id))
        
        conn.commit()
        conn.close()

# Global evolution engine
wade_evolution = WADEEvolutionEngine()
wade_teacher = WADETeachingInterface(wade_evolution)

def main():
    """Test the evolution engine"""
    
    # Example usage
    stats = wade_evolution.get_learning_stats()
    print("Learning Stats:", json.dumps(stats, indent=2, default=str))
    
    # Example strategy selection
    user_prompt = "Convert this Flask app to FastAPI"
    repo_analysis = {'frameworks': ['flask'], 'entry_points': ['app.py']}
    
    best_strategy = wade_evolution.select_best_strategy(user_prompt, repo_analysis)
    if best_strategy:
        print(f"Best strategy: {best_strategy.name} (Success rate: {best_strategy.success_rate:.2f})")
    
    # Example recommendations
    recommendations = wade_evolution.get_strategy_recommendations(user_prompt)
    print("Recommendations:", json.dumps(recommendations, indent=2))

if __name__ == "__main__":
    main()