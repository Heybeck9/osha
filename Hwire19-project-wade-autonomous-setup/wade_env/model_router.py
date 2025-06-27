#!/usr/bin/env python3
"""
WADE Model Router - Intelligent model selection based on task type
Automatically selects the most appropriate model for different types of tasks:
- Wizard for logic and reasoning
- Phind for code generation
- Qwen for summarization
- Claude for creative content
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import re

# Import WADE components
try:
    from settings_manager import settings_manager
    from model_manager import model_manager
    from performance import model_prewarmer, query_cache
    from security import request_signer
except ImportError:
    # For standalone testing
    from wade_env.settings_manager import settings_manager
    from wade_env.model_manager import model_manager
    from wade_env.performance import model_prewarmer, query_cache
    from wade_env.security import request_signer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_router")

class TaskType(Enum):
    """Enum for different types of tasks that require specialized models"""
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CREATIVE = "creative"
    GENERAL = "general"
    SECURITY = "security"
    DATA_ANALYSIS = "data_analysis"

class ModelRouter:
    """
    Intelligent model router that selects the most appropriate model for different task types.
    Can be used as middleware or directly via dispatch() method.
    """
    
    def __init__(self):
        """Initialize the model router with default settings"""
        self.settings = self._load_router_settings()
        self.model_specializations = self._get_model_specializations()
        self.task_patterns = self._get_task_patterns()
        self.override_model = None
        self.last_selected_model = None
        self.usage_stats = {model: 0 for model in self.model_specializations.keys()}
        self.performance_metrics = {
            "dispatch_times": [],
            "model_response_times": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Register models with the prewarmer
        for model in self.model_specializations.keys():
            model_prewarmer.register_model(model)
        
    def _load_router_settings(self) -> Dict[str, Any]:
        """Load router settings from settings manager"""
        try:
            router_settings = settings_manager.get_settings_dict().get("model_router", {})
            if not router_settings:
                # Initialize with defaults if not present
                router_settings = {
                    "enabled": True,
                    "override_model": None,
                    "model_specializations": {},
                    "task_patterns": {}
                }
                settings_manager.update_settings("model_router", router_settings)
            return router_settings
        except Exception as e:
            logger.error(f"Error loading router settings: {e}")
            return {
                "enabled": True,
                "override_model": None,
                "model_specializations": {},
                "task_patterns": {}
            }
    
    def _get_model_specializations(self) -> Dict[str, List[TaskType]]:
        """
        Get model specializations mapping from settings or use defaults
        Maps model names to their specialized task types
        """
        default_specializations = {
            "wizard-mega": [TaskType.REASONING, TaskType.GENERAL],
            "phind-codellama": [TaskType.CODE_GENERATION],
            "qwen-72b": [TaskType.SUMMARIZATION, TaskType.DATA_ANALYSIS],
            "claude-3-opus": [TaskType.CREATIVE, TaskType.REASONING],
            "llama3-70b": [TaskType.GENERAL],
            "mixtral-8x7b": [TaskType.GENERAL, TaskType.CODE_GENERATION],
            "gpt-4o": [TaskType.GENERAL, TaskType.REASONING, TaskType.CODE_GENERATION]
        }
        
        # Merge with any custom settings
        custom_specializations = self.settings.get("model_specializations", {})
        for model, specialties in custom_specializations.items():
            if model in default_specializations:
                # Convert string specialties to TaskType enum
                enum_specialties = [TaskType(s) if isinstance(s, str) else s for s in specialties]
                default_specializations[model] = enum_specialties
        
        return default_specializations
    
    def _get_task_patterns(self) -> Dict[TaskType, List[str]]:
        """
        Get regex patterns for identifying task types from user input
        Maps task types to lists of regex patterns
        """
        default_patterns = {
            TaskType.CODE_GENERATION: [
                r"(?i)write (a|some) code",
                r"(?i)implement (a|the)",
                r"(?i)create (a|the) function",
                r"(?i)generate (a|the) class",
                r"(?i)code (a|the|for)",
                r"(?i)develop (a|the)",
                r"(?i)program (a|the)",
                r"(?i)script (for|to)",
                r"(?i)fix (the|this) (code|bug)",
                r"(?i)debug (the|this)",
                r"(?i)refactor (the|this)",
                r"(?i)optimize (the|this) (code|function)",
                r"(?i)(html|css|javascript|python|java|c\+\+|typescript|go|rust)"
            ],
            TaskType.REASONING: [
                r"(?i)explain (why|how)",
                r"(?i)reason (about|through)",
                r"(?i)analyze (the|this)",
                r"(?i)evaluate (the|this)",
                r"(?i)compare (the|these)",
                r"(?i)what (is|are) (the|some)",
                r"(?i)why (is|does|would)",
                r"(?i)how (does|would|could)",
                r"(?i)solve (the|this) (problem|issue)",
                r"(?i)think (through|about)"
            ],
            TaskType.SUMMARIZATION: [
                r"(?i)summarize (the|this)",
                r"(?i)give (me|a) summary",
                r"(?i)condense (the|this)",
                r"(?i)tldr",
                r"(?i)brief (overview|summary)",
                r"(?i)key (points|takeaways)",
                r"(?i)main (ideas|points)",
                r"(?i)extract (the|important) information"
            ],
            TaskType.CREATIVE: [
                r"(?i)write (a|some) (story|poem|song|essay)",
                r"(?i)create (a|some) (content|story|narrative)",
                r"(?i)generate (a|some) (ideas|concepts)",
                r"(?i)brainstorm",
                r"(?i)imagine (a|some|if)",
                r"(?i)design (a|the|some)",
                r"(?i)creative (ideas|concepts)",
                r"(?i)invent (a|some)"
            ],
            TaskType.SECURITY: [
                r"(?i)security (audit|assessment|review)",
                r"(?i)vulnerability (scan|assessment)",
                r"(?i)penetration test",
                r"(?i)secure (the|this)",
                r"(?i)harden (the|this)",
                r"(?i)encryption",
                r"(?i)authentication",
                r"(?i)authorization",
                r"(?i)exploit",
                r"(?i)attack (vector|surface)",
                r"(?i)threat (model|assessment)"
            ],
            TaskType.DATA_ANALYSIS: [
                r"(?i)analyze (the|this) data",
                r"(?i)data (analysis|processing)",
                r"(?i)statistics",
                r"(?i)correlation",
                r"(?i)regression",
                r"(?i)machine learning",
                r"(?i)neural network",
                r"(?i)train (a|the) model",
                r"(?i)dataset",
                r"(?i)visualization",
                r"(?i)chart",
                r"(?i)graph",
                r"(?i)plot"
            ]
        }
        
        # Merge with any custom patterns
        custom_patterns = self.settings.get("task_patterns", {})
        for task_type, patterns in custom_patterns.items():
            task_enum = TaskType(task_type) if isinstance(task_type, str) else task_type
            if task_enum in default_patterns:
                default_patterns[task_enum].extend(patterns)
            else:
                default_patterns[task_enum] = patterns
        
        return default_patterns
    
    def set_override_model(self, model_name: Optional[str]) -> bool:
        """
        Set a model to override the automatic selection
        Returns True if successful, False otherwise
        """
        if model_name is None:
            self.override_model = None
            self.settings["override_model"] = None
            settings_manager.update_settings("model_router", self.settings)
            logger.info("Model override cleared")
            return True
        
        # Check if model exists
        available_models = model_manager.get_available_models()
        if model_name not in [m["name"] for m in available_models]:
            logger.error(f"Model {model_name} not available")
            return False
        
        self.override_model = model_name
        self.settings["override_model"] = model_name
        settings_manager.update_settings("model_router", self.settings)
        logger.info(f"Model override set to {model_name}")
        return True
    
    def is_enabled(self) -> bool:
        """Check if the model router is enabled"""
        return self.settings.get("enabled", True)
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the model router"""
        self.settings["enabled"] = enabled
        settings_manager.update_settings("model_router", self.settings)
        logger.info(f"Model router {'enabled' if enabled else 'disabled'}")
    
    def identify_task_type(self, prompt: str) -> TaskType:
        """
        Identify the type of task based on the prompt content
        Returns the most likely task type
        """
        task_matches = {}
        
        # Check each task type's patterns for matches
        for task_type, patterns in self.task_patterns.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, prompt):
                    match_count += 1
            
            if match_count > 0:
                task_matches[task_type] = match_count
        
        # Return the task type with the most matches, or GENERAL if none
        if task_matches:
            return max(task_matches.items(), key=lambda x: x[1])[0]
        return TaskType.GENERAL
    
    def select_model_for_task(self, task_type: TaskType) -> str:
        """
        Select the most appropriate model for a given task type
        Returns the name of the selected model
        """
        # Get available models
        available_models = model_manager.get_available_models()
        available_model_names = [m["name"] for m in available_models]
        
        # Find models specialized for this task type
        specialized_models = []
        for model, specialties in self.model_specializations.items():
            if task_type in specialties and model in available_model_names:
                specialized_models.append(model)
        
        if specialized_models:
            # Return the first available specialized model
            return specialized_models[0]
        
        # If no specialized model is available, return the first available model
        if available_model_names:
            return available_model_names[0]
        
        # Fallback to a default model
        return "gpt-3.5-turbo"
    
    async def dispatch(self, prompt: str) -> Tuple[str, TaskType]:
        """
        Main dispatch method - selects the appropriate model for a given prompt
        Returns the selected model name and identified task type
        """
        start_time = time.time()
        
        # Try to get from cache first
        cache_key = f"model_selection:{prompt}"
        cached_result = await query_cache.get(cache_key)
        if cached_result:
            self.performance_metrics["cache_hits"] += 1
            selected_model, task_type_value = cached_result
            task_type = TaskType(task_type_value)
            logger.info(f"Cache hit: Using model {selected_model} for task type {task_type.value}")
            return selected_model, task_type
        
        self.performance_metrics["cache_misses"] += 1
        
        if not self.is_enabled():
            # If router is disabled, use the current active model
            current_model = model_manager.get_active_model()
            return current_model, TaskType.GENERAL
        
        # If override is set, use that model
        if self.override_model:
            return self.override_model, TaskType.GENERAL
        
        # Identify task type from prompt
        task_type = self.identify_task_type(prompt)
        
        # Select appropriate model
        selected_model = self.select_model_for_task(task_type)
        
        # Ensure model is warmed up
        if not model_prewarmer.is_model_warmed(selected_model):
            logger.info(f"Warming up model {selected_model}...")
            await model_prewarmer.warmup_model(selected_model)
        
        # Update usage statistics
        self.usage_stats[selected_model] = self.usage_stats.get(selected_model, 0) + 1
        self.last_selected_model = selected_model
        
        # Set the selected model as active
        model_manager.set_active_model(selected_model)
        
        # Cache the result
        await query_cache.set(cache_key, (selected_model, task_type.value), ttl=3600)
        
        # Record performance metrics
        dispatch_time = time.time() - start_time
        self.performance_metrics["dispatch_times"].append(dispatch_time)
        
        logger.info(f"Selected model {selected_model} for task type {task_type.value} in {dispatch_time:.3f}s")
        return selected_model, task_type
        
    async def generate_with_model(self, model_name: str, prompt: str) -> str:
        """
        Generate a response using a specific model with performance tracking
        """
        start_time = time.time()
        
        # Sign the request for security
        payload = {"model": model_name, "prompt": prompt}
        signed_payload, signature = request_signer.sign_request(payload)
        
        try:
            # Use the model manager to generate a response
            response = await model_manager.generate(
                model_name=model_name,
                prompt=prompt,
                security_signature=signature
            )
            
            # Record performance metrics
            generation_time = time.time() - start_time
            if model_name not in self.performance_metrics["model_response_times"]:
                self.performance_metrics["model_response_times"][model_name] = []
            self.performance_metrics["model_response_times"][model_name].append(generation_time)
            
            logger.info(f"Generated response with {model_name} in {generation_time:.3f}s")
            return response
        except Exception as e:
            logger.error(f"Error generating with model {model_name}: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for models"""
        return self.usage_stats
    
    def get_last_selected_model(self) -> Optional[str]:
        """Get the last selected model"""
        return self.last_selected_model
    
    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.usage_stats = {model: 0 for model in self.model_specializations.keys()}
        self.last_selected_model = None
        self.performance_metrics = {
            "dispatch_times": [],
            "model_response_times": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the model router"""
        metrics = self.performance_metrics.copy()
        
        # Calculate average dispatch time
        if metrics["dispatch_times"]:
            metrics["avg_dispatch_time"] = sum(metrics["dispatch_times"]) / len(metrics["dispatch_times"])
        else:
            metrics["avg_dispatch_time"] = 0
        
        # Calculate average response time for each model
        avg_response_times = {}
        for model, times in metrics["model_response_times"].items():
            if times:
                avg_response_times[model] = sum(times) / len(times)
            else:
                avg_response_times[model] = 0
        metrics["avg_response_times"] = avg_response_times
        
        # Calculate cache hit rate
        total_requests = metrics["cache_hits"] + metrics["cache_misses"]
        if total_requests > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_requests
        else:
            metrics["cache_hit_rate"] = 0
        
        return metrics
    
    async def warmup_all_models(self) -> Dict[str, bool]:
        """Warm up all available models"""
        logger.info("Warming up all models...")
        return await model_prewarmer.warmup_all_models()

# Create singleton instance
model_router = ModelRouter()

# For testing
if __name__ == "__main__":
    async def test_router():
        prompts = [
            "Write a Python function to calculate Fibonacci numbers",
            "Explain how quantum computing works",
            "Summarize the key points of the latest climate report",
            "Write a short story about a robot that falls in love",
            "What is the capital of France?",
            "Perform a security audit on this code",
            "Analyze this dataset and create visualizations"
        ]
        
        for prompt in prompts:
            model, task_type = await model_router.dispatch(prompt)
            print(f"Prompt: {prompt}")
            print(f"Task Type: {task_type.value}")
            print(f"Selected Model: {model}")
            print("-" * 50)
    
    asyncio.run(test_router())