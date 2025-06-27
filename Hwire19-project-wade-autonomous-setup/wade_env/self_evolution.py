#!/usr/bin/env python3
"""
WADE Self Evolution - Autonomous system improvement
Tracks performance, collects feedback, and evolves capabilities over time
"""

import os
import json
import logging
import asyncio
import time
import random
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import re

# Import WADE components
try:
    from settings_manager import settings_manager
    from file_manager import file_manager
except ImportError:
    # For standalone testing
    from wade_env.settings_manager import settings_manager
    from wade_env.file_manager import file_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self_evolution")

class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SUGGESTION = "suggestion"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"

class EvolutionStage(Enum):
    """Stages of the evolution process"""
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_id: str
    name: str
    value: float
    timestamp: float
    category: str
    description: Optional[str] = None
    unit: Optional[str] = None
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    is_critical: bool = False

@dataclass
class Feedback:
    """User feedback data structure"""
    feedback_id: str
    type: FeedbackType
    content: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    rating: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    processed: bool = False

@dataclass
class EvolutionTask:
    """Evolution task data structure"""
    task_id: str
    name: str
    description: str
    priority: int  # 1-5, with 5 being highest
    status: str  # "pending", "in_progress", "completed", "failed"
    created_at: float
    updated_at: float
    stage: EvolutionStage
    metrics_affected: List[str] = field(default_factory=list)
    feedback_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    implementation_plan: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class SelfEvolution:
    """
    Self-evolution manager for WADE
    Tracks performance, collects feedback, and evolves capabilities over time
    """
    
    def __init__(self):
        """Initialize the self-evolution manager"""
        self.settings = self._load_evolution_settings()
        self.data_dir = self.settings.get("data_dir", "/workspace/wade_env/evolution_data")
        self.version = self.settings.get("wade_logic_version", 1)
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.feedback: Dict[str, Feedback] = {}
        self.tasks: Dict[str, EvolutionTask] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        self.current_evolution: Optional[Dict[str, Any]] = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing data
        self._load_data()
    
    def _load_evolution_settings(self) -> Dict[str, Any]:
        """Load evolution settings from settings manager"""
        try:
            evolution_settings = settings_manager.get_settings_dict().get("self_evolution", {})
            if not evolution_settings:
                # Initialize with defaults if not present
                evolution_settings = {
                    "enabled": True,
                    "data_dir": "/workspace/wade_env/evolution_data",
                    "auto_evolution_enabled": False,
                    "evolution_threshold": 10,  # Number of feedback items to trigger evolution
                    "evolution_interval_days": 7,
                    "wade_logic_version": 1,
                    "metrics_retention_days": 90,
                    "feedback_retention_days": 180,
                    "critical_metrics": [
                        "response_time",
                        "task_success_rate",
                        "user_satisfaction"
                    ]
                }
                settings_manager.update_settings("self_evolution", evolution_settings)
            return evolution_settings
        except Exception as e:
            logger.error(f"Error loading evolution settings: {e}")
            return {
                "enabled": True,
                "data_dir": "/workspace/wade_env/evolution_data",
                "auto_evolution_enabled": False,
                "evolution_threshold": 10,
                "evolution_interval_days": 7,
                "wade_logic_version": 1,
                "metrics_retention_days": 90,
                "feedback_retention_days": 180,
                "critical_metrics": [
                    "response_time",
                    "task_success_rate",
                    "user_satisfaction"
                ]
            }
    
    def _load_data(self) -> None:
        """Load existing data from files"""
        try:
            # Load metrics
            metrics_path = os.path.join(self.data_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    for metric_id, metric_data in metrics_data.items():
                        self.metrics[metric_id] = PerformanceMetric(**metric_data)
            
            # Load feedback
            feedback_path = os.path.join(self.data_dir, "feedback.json")
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    feedback_data = json.load(f)
                    for feedback_id, feedback_item in feedback_data.items():
                        # Convert string enum to actual enum
                        feedback_item["type"] = FeedbackType(feedback_item["type"])
                        self.feedback[feedback_id] = Feedback(**feedback_item)
            
            # Load tasks
            tasks_path = os.path.join(self.data_dir, "tasks.json")
            if os.path.exists(tasks_path):
                with open(tasks_path, 'r') as f:
                    tasks_data = json.load(f)
                    for task_id, task_data in tasks_data.items():
                        # Convert string enum to actual enum
                        task_data["stage"] = EvolutionStage(task_data["stage"])
                        self.tasks[task_id] = EvolutionTask(**task_data)
            
            # Load evolution history
            history_path = os.path.join(self.data_dir, "evolution_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.evolution_history = json.load(f)
            
            logger.info(f"Loaded {len(self.metrics)} metrics, {len(self.feedback)} feedback items, "
                      f"{len(self.tasks)} tasks, and {len(self.evolution_history)} evolution records")
        except Exception as e:
            logger.error(f"Error loading evolution data: {e}")
    
    def _save_data(self) -> None:
        """Save data to files"""
        try:
            # Save metrics
            metrics_data = {metric_id: asdict(metric) for metric_id, metric in self.metrics.items()}
            metrics_path = os.path.join(self.data_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save feedback
            feedback_data = {}
            for feedback_id, feedback_item in self.feedback.items():
                feedback_dict = asdict(feedback_item)
                # Convert enum to string for JSON serialization
                feedback_dict["type"] = feedback_dict["type"].value
                feedback_data[feedback_id] = feedback_dict
            
            feedback_path = os.path.join(self.data_dir, "feedback.json")
            with open(feedback_path, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            # Save tasks
            tasks_data = {}
            for task_id, task in self.tasks.items():
                task_dict = asdict(task)
                # Convert enum to string for JSON serialization
                task_dict["stage"] = task_dict["stage"].value
                tasks_data[task_id] = task_dict
            
            tasks_path = os.path.join(self.data_dir, "tasks.json")
            with open(tasks_path, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            # Save evolution history
            history_path = os.path.join(self.data_dir, "evolution_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.evolution_history, f, indent=2)
            
            logger.info("Evolution data saved successfully")
        except Exception as e:
            logger.error(f"Error saving evolution data: {e}")
    
    def record_metric(self, name: str, value: float, category: str, 
                     description: Optional[str] = None, unit: Optional[str] = None,
                     threshold_min: Optional[float] = None, 
                     threshold_max: Optional[float] = None,
                     is_critical: bool = False) -> str:
        """
        Record a performance metric
        Returns the ID of the recorded metric
        """
        timestamp = time.time()
        metric_id = f"metric_{hashlib.md5(f'{name}_{category}_{timestamp}'.encode()).hexdigest()[:12]}"
        
        metric = PerformanceMetric(
            metric_id=metric_id,
            name=name,
            value=value,
            timestamp=timestamp,
            category=category,
            description=description,
            unit=unit,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            is_critical=is_critical
        )
        
        self.metrics[metric_id] = metric
        
        # Check if this metric exceeds thresholds
        if self._check_metric_thresholds(metric):
            # If it's a critical metric and exceeds thresholds, trigger analysis
            if is_critical and self.settings.get("auto_evolution_enabled", False):
                asyncio.create_task(self.analyze_performance())
        
        # Save data
        self._save_data()
        
        return metric_id
    
    def _check_metric_thresholds(self, metric: PerformanceMetric) -> bool:
        """
        Check if a metric exceeds its thresholds
        Returns True if thresholds are exceeded
        """
        if metric.threshold_min is not None and metric.value < metric.threshold_min:
            logger.warning(f"Metric {metric.name} below minimum threshold: {metric.value} < {metric.threshold_min}")
            return True
        
        if metric.threshold_max is not None and metric.value > metric.threshold_max:
            logger.warning(f"Metric {metric.name} above maximum threshold: {metric.value} > {metric.threshold_max}")
            return True
        
        return False
    
    def record_feedback(self, content: str, feedback_type: Union[FeedbackType, str], 
                       context: Optional[Dict[str, Any]] = None,
                       user_id: Optional[str] = None,
                       rating: Optional[int] = None,
                       tags: Optional[List[str]] = None) -> str:
        """
        Record user feedback
        Returns the ID of the recorded feedback
        """
        timestamp = time.time()
        feedback_id = f"feedback_{hashlib.md5(f'{content}_{timestamp}'.encode()).hexdigest()[:12]}"
        
        # Convert string to enum if needed
        if isinstance(feedback_type, str):
            try:
                feedback_type = FeedbackType(feedback_type)
            except ValueError:
                feedback_type = FeedbackType.NEUTRAL
        
        feedback = Feedback(
            feedback_id=feedback_id,
            type=feedback_type,
            content=content,
            timestamp=timestamp,
            context=context or {},
            user_id=user_id,
            rating=rating,
            tags=tags or [],
            processed=False
        )
        
        self.feedback[feedback_id] = feedback
        
        # Check if we've reached the threshold for auto-evolution
        if (self.settings.get("auto_evolution_enabled", False) and
            len([f for f in self.feedback.values() if not f.processed]) >= 
            self.settings.get("evolution_threshold", 10)):
            asyncio.create_task(self.evolve())
        
        # Save data
        self._save_data()
        
        return feedback_id
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze system performance based on metrics and feedback
        Returns analysis results
        """
        logger.info("Starting performance analysis")
        
        # Get recent metrics (last 7 days)
        recent_cutoff = time.time() - (7 * 24 * 60 * 60)
        recent_metrics = [m for m in self.metrics.values() if m.timestamp >= recent_cutoff]
        
        # Get unprocessed feedback
        unprocessed_feedback = [f for f in self.feedback.values() if not f.processed]
        
        # Group metrics by category
        metrics_by_category = {}
        for metric in recent_metrics:
            if metric.category not in metrics_by_category:
                metrics_by_category[metric.category] = []
            metrics_by_category[metric.category].append(metric)
        
        # Calculate statistics for each category
        category_stats = {}
        for category, metrics in metrics_by_category.items():
            values = [m.value for m in metrics]
            if not values:
                continue
            
            category_stats[category] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "metrics": [m.name for m in metrics]
            }
        
        # Analyze feedback sentiment
        feedback_sentiment = {
            "positive": len([f for f in unprocessed_feedback if f.type == FeedbackType.POSITIVE]),
            "negative": len([f for f in unprocessed_feedback if f.type == FeedbackType.NEGATIVE]),
            "neutral": len([f for f in unprocessed_feedback if f.type == FeedbackType.NEUTRAL]),
            "suggestions": len([f for f in unprocessed_feedback if f.type == FeedbackType.SUGGESTION]),
            "bugs": len([f for f in unprocessed_feedback if f.type == FeedbackType.BUG]),
            "feature_requests": len([f for f in unprocessed_feedback if f.type == FeedbackType.FEATURE_REQUEST])
        }
        
        # Extract common themes from feedback
        feedback_themes = await self._extract_feedback_themes(unprocessed_feedback)
        
        # Identify areas for improvement
        improvement_areas = []
        
        # Check metrics against thresholds
        for metric in recent_metrics:
            if self._check_metric_thresholds(metric):
                improvement_areas.append({
                    "type": "metric",
                    "name": metric.name,
                    "category": metric.category,
                    "current_value": metric.value,
                    "threshold_min": metric.threshold_min,
                    "threshold_max": metric.threshold_max,
                    "priority": 5 if metric.is_critical else 3
                })
        
        # Add areas from feedback themes
        for theme, count in feedback_themes.items():
            if count >= 2:  # Only consider themes mentioned multiple times
                priority = 5 if count >= 5 else (4 if count >= 3 else 3)
                improvement_areas.append({
                    "type": "feedback_theme",
                    "name": theme,
                    "count": count,
                    "priority": priority
                })
        
        # Sort improvement areas by priority
        improvement_areas.sort(key=lambda x: x["priority"], reverse=True)
        
        analysis_results = {
            "timestamp": time.time(),
            "metrics_analyzed": len(recent_metrics),
            "feedback_analyzed": len(unprocessed_feedback),
            "category_stats": category_stats,
            "feedback_sentiment": feedback_sentiment,
            "feedback_themes": feedback_themes,
            "improvement_areas": improvement_areas
        }
        
        logger.info(f"Performance analysis complete: identified {len(improvement_areas)} improvement areas")
        return analysis_results
    
    async def _extract_feedback_themes(self, feedback_items: List[Feedback]) -> Dict[str, int]:
        """
        Extract common themes from feedback
        Returns a dictionary of themes and their frequencies
        """
        # This would ideally use NLP, but for simplicity we'll use keyword matching
        themes = {}
        
        # Define common themes and their keywords
        theme_keywords = {
            "performance": ["slow", "fast", "speed", "performance", "lag", "latency"],
            "usability": ["usability", "user-friendly", "intuitive", "confusing", "difficult", "easy"],
            "reliability": ["crash", "error", "bug", "reliable", "stability", "unstable"],
            "features": ["feature", "functionality", "capability", "ability", "support for"],
            "ui": ["interface", "ui", "design", "layout", "look", "feel", "appearance"],
            "documentation": ["docs", "documentation", "help", "tutorial", "guide", "example"],
            "integration": ["integration", "connect", "api", "plugin", "extension"]
        }
        
        # Count occurrences of themes in feedback
        for feedback in feedback_items:
            content_lower = feedback.content.lower()
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    themes[theme] = themes.get(theme, 0) + 1
        
        return themes
    
    async def create_evolution_tasks(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Create evolution tasks based on analysis results
        Returns a list of created task IDs
        """
        task_ids = []
        
        # Create tasks for each improvement area
        for area in analysis_results.get("improvement_areas", []):
            task_id = f"task_{hashlib.md5(f'{area['name']}_{time.time()}'.encode()).hexdigest()[:12]}"
            
            # Determine task description based on area type
            if area["type"] == "metric":
                description = f"Improve {area['name']} metric in {area['category']} category. "
                if area.get("threshold_min") is not None and area["current_value"] < area["threshold_min"]:
                    description += f"Current value ({area['current_value']}) is below minimum threshold ({area['threshold_min']})."
                elif area.get("threshold_max") is not None and area["current_value"] > area["threshold_max"]:
                    description += f"Current value ({area['current_value']}) is above maximum threshold ({area['threshold_max']})."
            else:  # feedback_theme
                description = f"Address feedback theme: {area['name']} (mentioned {area['count']} times)"
            
            task = EvolutionTask(
                task_id=task_id,
                name=f"Improve {area['name']}",
                description=description,
                priority=area["priority"],
                status="pending",
                created_at=time.time(),
                updated_at=time.time(),
                stage=EvolutionStage.PLANNING
            )
            
            self.tasks[task_id] = task
            task_ids.append(task_id)
        
        # Save data
        self._save_data()
        
        logger.info(f"Created {len(task_ids)} evolution tasks")
        return task_ids
    
    async def generate_implementation_plan(self, task_id: str) -> Optional[str]:
        """
        Generate an implementation plan for a task
        Returns the implementation plan as a string
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return None
        
        task = self.tasks[task_id]
        
        # In a real implementation, this would use an LLM to generate a plan
        # For now, we'll use a template-based approach
        
        plan = f"# Implementation Plan for: {task.name}\n\n"
        plan += f"## Description\n{task.description}\n\n"
        plan += "## Steps\n\n"
        
        if "metric" in task.name.lower() or "performance" in task.name.lower():
            plan += "1. Analyze current implementation to identify bottlenecks\n"
            plan += "2. Profile code execution to pinpoint slow operations\n"
            plan += "3. Implement optimizations for identified bottlenecks\n"
            plan += "4. Add caching mechanisms where appropriate\n"
            plan += "5. Refactor code for better performance\n"
            plan += "6. Test changes to ensure they improve the metric\n"
            plan += "7. Deploy and monitor results\n"
        elif "usability" in task.name.lower() or "ui" in task.name.lower():
            plan += "1. Review user feedback related to interface issues\n"
            plan += "2. Identify specific UI elements causing confusion\n"
            plan += "3. Create mockups for improved interface\n"
            plan += "4. Implement UI changes\n"
            plan += "5. Conduct usability testing\n"
            plan += "6. Refine based on testing results\n"
            plan += "7. Deploy and collect new feedback\n"
        elif "reliability" in task.name.lower() or "bug" in task.name.lower():
            plan += "1. Reproduce reported issues consistently\n"
            plan += "2. Add tests that verify the bug\n"
            plan += "3. Debug to identify root causes\n"
            plan += "4. Implement fixes\n"
            plan += "5. Add error handling and recovery mechanisms\n"
            plan += "6. Verify fixes with tests\n"
            plan += "7. Deploy and monitor for regressions\n"
        elif "feature" in task.name.lower():
            plan += "1. Define detailed requirements for the feature\n"
            plan += "2. Design the feature architecture\n"
            plan += "3. Create implementation plan with milestones\n"
            plan += "4. Implement core functionality\n"
            plan += "5. Add tests for the new feature\n"
            plan += "6. Document the feature\n"
            plan += "7. Deploy and announce to users\n"
        else:
            plan += "1. Analyze current implementation\n"
            plan += "2. Identify specific areas for improvement\n"
            plan += "3. Design changes\n"
            plan += "4. Implement changes\n"
            plan += "5. Test thoroughly\n"
            plan += "6. Deploy and monitor\n"
        
        plan += "\n## Success Criteria\n"
        plan += "- Specific measurable improvements in relevant metrics\n"
        plan += "- Positive user feedback\n"
        plan += "- No regressions in other functionality\n"
        
        # Update the task with the plan
        task.implementation_plan = plan
        task.updated_at = time.time()
        task.stage = EvolutionStage.IMPLEMENTATION
        
        # Save data
        self._save_data()
        
        return plan
    
    async def evolve(self) -> Dict[str, Any]:
        """
        Execute the evolution process
        Returns metadata about the evolution
        """
        if self.current_evolution:
            return {
                "success": False,
                "message": "Evolution already in progress",
                "current_evolution": self.current_evolution
            }
        
        logger.info("Starting evolution process")
        
        # Create a new evolution record
        evolution_id = f"evolution_{int(time.time())}"
        self.current_evolution = {
            "evolution_id": evolution_id,
            "version_from": self.version,
            "version_to": self.version + 1,
            "started_at": time.time(),
            "status": "in_progress",
            "stages": [],
            "metrics_before": {},
            "tasks": []
        }
        
        try:
            # Stage 1: Analysis
            self.current_evolution["stages"].append({
                "stage": "analysis",
                "started_at": time.time(),
                "status": "in_progress"
            })
            
            # Record current metrics for comparison
            critical_metrics = self.settings.get("critical_metrics", [])
            for metric_name in critical_metrics:
                matching_metrics = [m for m in self.metrics.values() if m.name == metric_name]
                if matching_metrics:
                    # Get the most recent metric
                    latest_metric = max(matching_metrics, key=lambda m: m.timestamp)
                    self.current_evolution["metrics_before"][metric_name] = latest_metric.value
            
            # Perform analysis
            analysis_results = await self.analyze_performance()
            
            # Update stage status
            self.current_evolution["stages"][-1].update({
                "completed_at": time.time(),
                "status": "completed",
                "results": {
                    "improvement_areas": len(analysis_results["improvement_areas"]),
                    "metrics_analyzed": analysis_results["metrics_analyzed"],
                    "feedback_analyzed": analysis_results["feedback_analyzed"]
                }
            })
            
            # Stage 2: Planning
            self.current_evolution["stages"].append({
                "stage": "planning",
                "started_at": time.time(),
                "status": "in_progress"
            })
            
            # Create tasks based on analysis
            task_ids = await self.create_evolution_tasks(analysis_results)
            self.current_evolution["tasks"] = task_ids
            
            # Generate implementation plans for each task
            for task_id in task_ids:
                await self.generate_implementation_plan(task_id)
            
            # Update stage status
            self.current_evolution["stages"][-1].update({
                "completed_at": time.time(),
                "status": "completed",
                "results": {
                    "tasks_created": len(task_ids)
                }
            })
            
            # Stage 3: Implementation
            self.current_evolution["stages"].append({
                "stage": "implementation",
                "started_at": time.time(),
                "status": "in_progress"
            })
            
            # In a real implementation, this would execute the tasks
            # For now, we'll simulate task execution
            await self._simulate_task_execution(task_ids)
            
            # Update stage status
            self.current_evolution["stages"][-1].update({
                "completed_at": time.time(),
                "status": "completed",
                "results": {
                    "tasks_completed": len([t for t in task_ids if self.tasks[t].status == "completed"]),
                    "tasks_failed": len([t for t in task_ids if self.tasks[t].status == "failed"])
                }
            })
            
            # Stage 4: Deployment
            self.current_evolution["stages"].append({
                "stage": "deployment",
                "started_at": time.time(),
                "status": "in_progress"
            })
            
            # Increment version
            self.version += 1
            self.settings["wade_logic_version"] = self.version
            settings_manager.update_settings("self_evolution", self.settings)
            
            # Mark feedback as processed
            for feedback_id, feedback in self.feedback.items():
                if not feedback.processed:
                    feedback.processed = True
            
            # Update stage status
            self.current_evolution["stages"][-1].update({
                "completed_at": time.time(),
                "status": "completed",
                "results": {
                    "new_version": self.version
                }
            })
            
            # Finalize evolution
            self.current_evolution.update({
                "completed_at": time.time(),
                "status": "completed",
                "duration_seconds": time.time() - self.current_evolution["started_at"]
            })
            
            # Add to history
            self.evolution_history.append(self.current_evolution)
            
            # Save data
            self._save_data()
            
            logger.info(f"Evolution completed: version {self.version}")
            
            result = {
                "success": True,
                "evolution_id": evolution_id,
                "version": self.version,
                "tasks_completed": len([t for t in task_ids if self.tasks[t].status == "completed"]),
                "duration_seconds": self.current_evolution["duration_seconds"]
            }
            
            # Clear current evolution
            current_evolution = self.current_evolution
            self.current_evolution = None
            
            return result
        
        except Exception as e:
            logger.error(f"Error during evolution: {e}")
            
            # Mark evolution as failed
            if self.current_evolution:
                self.current_evolution.update({
                    "completed_at": time.time(),
                    "status": "failed",
                    "error": str(e),
                    "duration_seconds": time.time() - self.current_evolution["started_at"]
                })
                
                # Add to history
                self.evolution_history.append(self.current_evolution)
                
                # Save data
                self._save_data()
                
                # Clear current evolution
                self.current_evolution = None
            
            return {
                "success": False,
                "message": f"Evolution failed: {str(e)}"
            }
    
    async def _simulate_task_execution(self, task_ids: List[str]) -> None:
        """Simulate execution of evolution tasks (for demonstration)"""
        for task_id in task_ids:
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            
            # Simulate task execution time based on priority
            execution_time = 1 + (5 - task.priority) * 0.5
            await asyncio.sleep(execution_time)
            
            # 80% chance of success
            if random.random() < 0.8:
                task.status = "completed"
                task.result = {
                    "success": True,
                    "message": "Task completed successfully",
                    "changes_made": random.randint(1, 5)
                }
            else:
                task.status = "failed"
                task.result = {
                    "success": False,
                    "message": "Task failed due to unexpected error"
                }
            
            task.updated_at = time.time()
    
    def get_version(self) -> int:
        """Get the current WADE logic version"""
        return self.version
    
    def get_metrics(self, category: Optional[str] = None, 
                   days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get performance metrics
        Optionally filtered by category and time range
        """
        filtered_metrics = []
        
        # Apply time filter if specified
        if days is not None:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
        else:
            cutoff_time = 0
        
        for metric in self.metrics.values():
            if metric.timestamp >= cutoff_time:
                if category is None or metric.category == category:
                    filtered_metrics.append(asdict(metric))
        
        # Sort by timestamp (newest first)
        filtered_metrics.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return filtered_metrics
    
    def get_feedback(self, feedback_type: Optional[Union[FeedbackType, str]] = None,
                    processed: Optional[bool] = None,
                    days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get user feedback
        Optionally filtered by type, processed status, and time range
        """
        filtered_feedback = []
        
        # Convert string to enum if needed
        if isinstance(feedback_type, str):
            try:
                feedback_type = FeedbackType(feedback_type)
            except ValueError:
                feedback_type = None
        
        # Apply time filter if specified
        if days is not None:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
        else:
            cutoff_time = 0
        
        for feedback in self.feedback.values():
            if feedback.timestamp >= cutoff_time:
                if (feedback_type is None or feedback.type == feedback_type) and \
                   (processed is None or feedback.processed == processed):
                    feedback_dict = asdict(feedback)
                    # Convert enum to string for JSON serialization
                    feedback_dict["type"] = feedback_dict["type"].value
                    filtered_feedback.append(feedback_dict)
        
        # Sort by timestamp (newest first)
        filtered_feedback.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return filtered_feedback
    
    def get_tasks(self, status: Optional[str] = None,
                 priority: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get evolution tasks
        Optionally filtered by status and priority
        """
        filtered_tasks = []
        
        for task in self.tasks.values():
            if (status is None or task.status == status) and \
               (priority is None or task.priority == priority):
                task_dict = asdict(task)
                # Convert enum to string for JSON serialization
                task_dict["stage"] = task_dict["stage"].value
                filtered_tasks.append(task_dict)
        
        # Sort by priority (highest first) and then by created_at (newest first)
        filtered_tasks.sort(key=lambda x: (-x["priority"], -x["created_at"]))
        
        return filtered_tasks
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get the history of evolution processes"""
        return self.evolution_history
    
    def get_current_evolution(self) -> Optional[Dict[str, Any]]:
        """Get the currently running evolution process, if any"""
        return self.current_evolution
    
    def is_evolution_in_progress(self) -> bool:
        """Check if an evolution process is currently running"""
        return self.current_evolution is not None
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """
        Clean up old metrics and feedback data
        Returns counts of removed items
        """
        metrics_retention_days = self.settings.get("metrics_retention_days", 90)
        feedback_retention_days = self.settings.get("feedback_retention_days", 180)
        
        metrics_cutoff = time.time() - (metrics_retention_days * 24 * 60 * 60)
        feedback_cutoff = time.time() - (feedback_retention_days * 24 * 60 * 60)
        
        # Remove old metrics
        old_metrics = [m_id for m_id, m in self.metrics.items() if m.timestamp < metrics_cutoff]
        for metric_id in old_metrics:
            del self.metrics[metric_id]
        
        # Remove old feedback
        old_feedback = [f_id for f_id, f in self.feedback.items() if f.timestamp < feedback_cutoff]
        for feedback_id in old_feedback:
            del self.feedback[feedback_id]
        
        # Save data
        self._save_data()
        
        return {
            "metrics_removed": len(old_metrics),
            "feedback_removed": len(old_feedback)
        }

# Create singleton instance
self_evolution = SelfEvolution()

# For testing
if __name__ == "__main__":
    async def test_self_evolution():
        # Record some test metrics
        self_evolution.record_metric("response_time", 250, "performance", 
                                    description="API response time in ms", 
                                    unit="ms", threshold_max=200, is_critical=True)
        
        self_evolution.record_metric("task_success_rate", 0.85, "reliability",
                                    description="Percentage of tasks completed successfully",
                                    unit="%", threshold_min=0.9, is_critical=True)
        
        # Record some test feedback
        self_evolution.record_feedback("The system is too slow when processing large files",
                                     FeedbackType.NEGATIVE, 
                                     context={"file_size": "25MB"},
                                     rating=2)
        
        self_evolution.record_feedback("I love the new UI, much more intuitive!",
                                     FeedbackType.POSITIVE,
                                     rating=5)
        
        # Trigger evolution
        evolution_result = await self_evolution.evolve()
        print("Evolution result:", evolution_result)
        
        # Get metrics
        metrics = self_evolution.get_metrics(category="performance")
        print(f"Performance metrics: {len(metrics)}")
        
        # Get tasks
        tasks = self_evolution.get_tasks()
        print(f"Evolution tasks: {len(tasks)}")
        
        # Get version
        version = self_evolution.get_version()
        print(f"Current version: {version}")
    
    asyncio.run(test_self_evolution())