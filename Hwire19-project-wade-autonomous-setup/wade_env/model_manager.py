#!/usr/bin/env python3
"""
WADE Model Manager - LLM configuration and management
Handles multiple model providers and adaptive model selection
"""

import json
import asyncio
import aiohttp
import subprocess
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    name: str
    provider: str  # "ollama", "openai", "anthropic", "local"
    model_id: str
    api_endpoint: str
    capabilities: List[str]
    context_length: int
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    enabled: bool = True
    priority: int = 1  # Higher = preferred

@dataclass
class ModelResponse:
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class ModelManager:
    def __init__(self, config_path: str = "/workspace/wade_env/model_config.yaml"):
        self.config_path = Path(config_path)
        self.models: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.fallback_models: List[str] = []
        self.load_config()
    
    def load_config(self):
        """Load model configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Load models
                for model_data in config_data.get('models', []):
                    model = ModelConfig(**model_data)
                    self.models[model.name] = model
                
                # Set active model
                self.active_model = config_data.get('active_model')
                self.fallback_models = config_data.get('fallback_models', [])
                
                logging.info(f"Loaded {len(self.models)} model configurations")
                
            except Exception as e:
                logging.error(f"Error loading model config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default model configuration"""
        default_models = [
            ModelConfig(
                name="phind-codellama",
                provider="ollama",
                model_id="phind-codellama:latest",
                api_endpoint="http://localhost:11434",
                capabilities=["code", "reasoning", "analysis"],
                context_length=16384,
                temperature=0.1,
                max_tokens=4096,
                priority=10
            ),
            ModelConfig(
                name="deepseek-coder",
                provider="ollama", 
                model_id="deepseek-coder:6.7b",
                api_endpoint="http://localhost:11434",
                capabilities=["code", "debugging", "optimization"],
                context_length=8192,
                temperature=0.2,
                max_tokens=2048,
                priority=8
            ),
            ModelConfig(
                name="wizardlm",
                provider="ollama",
                model_id="wizardlm:7b",
                api_endpoint="http://localhost:11434", 
                capabilities=["reasoning", "planning", "creative"],
                context_length=4096,
                temperature=0.7,
                max_tokens=2048,
                priority=6
            ),
            ModelConfig(
                name="llama2",
                provider="ollama",
                model_id="llama2:7b",
                api_endpoint="http://localhost:11434",
                capabilities=["general", "conversation"],
                context_length=4096,
                temperature=0.7,
                max_tokens=2048,
                priority=4
            )
        ]
        
        for model in default_models:
            self.models[model.name] = model
        
        self.active_model = "phind-codellama"
        self.fallback_models = ["deepseek-coder", "wizardlm", "llama2"]
        
        self.save_config()
        logging.info("Created default model configuration")
    
    def save_config(self):
        """Save model configuration to file"""
        try:
            config_data = {
                'active_model': self.active_model,
                'fallback_models': self.fallback_models,
                'models': [asdict(model) for model in self.models.values()]
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logging.info("Model configuration saved")
            
        except Exception as e:
            logging.error(f"Error saving model config: {e}")
    
    async def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available and responsive"""
        if model_name not in self.models:
            return False
        
        model = self.models[model_name]
        
        try:
            if model.provider == "ollama":
                return await self._check_ollama_model(model)
            elif model.provider == "openai":
                return await self._check_openai_model(model)
            else:
                return False
                
        except Exception as e:
            logging.error(f"Error checking model {model_name}: {e}")
            return False
    
    async def _check_ollama_model(self, model: ModelConfig) -> bool:
        """Check Ollama model availability"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get(f"{model.api_endpoint}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        return model.model_id in models
            return False
            
        except Exception:
            return False
    
    async def _check_openai_model(self, model: ModelConfig) -> bool:
        """Check OpenAI model availability"""
        try:
            headers = {"Authorization": f"Bearer {model.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{model.api_endpoint}/v1/models",
                    headers=headers,
                    timeout=5
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def pull_ollama_model(self, model_name: str) -> bool:
        """Pull/download an Ollama model"""
        if model_name not in self.models:
            return False
        
        model = self.models[model_name]
        if model.provider != "ollama":
            return False
        
        try:
            logging.info(f"Pulling Ollama model: {model.model_id}")
            
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model.model_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logging.info(f"Successfully pulled model: {model.model_id}")
                return True
            else:
                logging.error(f"Failed to pull model: {stderr.decode()}")
                return False
                
        except Exception as e:
            logging.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate_response(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> ModelResponse:
        """Generate response using specified or active model"""
        target_model = model_name or self.active_model
        
        if not target_model or target_model not in self.models:
            return ModelResponse(
                content="",
                model_used="none",
                tokens_used=0,
                response_time=0,
                success=False,
                error="No valid model specified"
            )
        
        # Try primary model first
        response = await self._try_model(target_model, prompt, **kwargs)
        if response.success:
            return response
        
        # Try fallback models
        for fallback in self.fallback_models:
            if fallback != target_model and fallback in self.models:
                logging.info(f"Trying fallback model: {fallback}")
                response = await self._try_model(fallback, prompt, **kwargs)
                if response.success:
                    return response
        
        return ModelResponse(
            content="",
            model_used="none",
            tokens_used=0,
            response_time=0,
            success=False,
            error="All models failed to respond"
        )
    
    async def _try_model(self, model_name: str, prompt: str, **kwargs) -> ModelResponse:
        """Try to generate response with specific model"""
        import time
        start_time = time.time()
        
        model = self.models[model_name]
        
        try:
            if model.provider == "ollama":
                return await self._ollama_generate(model, prompt, **kwargs)
            elif model.provider == "openai":
                return await self._openai_generate(model, prompt, **kwargs)
            else:
                return ModelResponse(
                    content="",
                    model_used=model_name,
                    tokens_used=0,
                    response_time=time.time() - start_time,
                    success=False,
                    error=f"Unsupported provider: {model.provider}"
                )
                
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def _ollama_generate(self, model: ModelConfig, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Ollama"""
        import time
        start_time = time.time()
        
        payload = {
            "model": model.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", model.temperature),
                "num_predict": kwargs.get("max_tokens", model.max_tokens)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{model.api_endpoint}/api/generate",
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ModelResponse(
                        content=data.get("response", ""),
                        model_used=model.name,
                        tokens_used=data.get("eval_count", 0),
                        response_time=time.time() - start_time,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return ModelResponse(
                        content="",
                        model_used=model.name,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
    
    async def _openai_generate(self, model: ModelConfig, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using OpenAI API"""
        import time
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", model.temperature),
            "max_tokens": kwargs.get("max_tokens", model.max_tokens)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{model.api_endpoint}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    choice = data["choices"][0]
                    
                    return ModelResponse(
                        content=choice["message"]["content"],
                        model_used=model.name,
                        tokens_used=data["usage"]["total_tokens"],
                        response_time=time.time() - start_time,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return ModelResponse(
                        content="",
                        model_used=model.name,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
    
    def add_model(self, model: ModelConfig):
        """Add a new model configuration"""
        self.models[model.name] = model
        self.save_config()
        logging.info(f"Added model: {model.name}")
    
    def remove_model(self, model_name: str):
        """Remove a model configuration"""
        if model_name in self.models:
            del self.models[model_name]
            
            # Update active model if removed
            if self.active_model == model_name:
                available_models = [name for name, model in self.models.items() if model.enabled]
                self.active_model = available_models[0] if available_models else None
            
            # Update fallback models
            if model_name in self.fallback_models:
                self.fallback_models.remove(model_name)
            
            self.save_config()
            logging.info(f"Removed model: {model_name}")
    
    def set_active_model(self, model_name: str):
        """Set the active model"""
        if model_name in self.models:
            self.active_model = model_name
            self.save_config()
            logging.info(f"Set active model: {model_name}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return [
            {
                "name": model.name,
                "provider": model.provider,
                "model_id": model.model_id,
                "capabilities": model.capabilities,
                "enabled": model.enabled,
                "active": model.name == self.active_model
            }
            for model in self.models.values()
        ]
    
    async def auto_select_model(self, task_type: str) -> Optional[str]:
        """Automatically select best model for task type"""
        capability_map = {
            "code": ["code", "reasoning", "analysis"],
            "debug": ["code", "debugging", "analysis"],
            "creative": ["creative", "reasoning"],
            "analysis": ["analysis", "reasoning"],
            "general": ["general", "conversation"]
        }
        
        required_capabilities = capability_map.get(task_type, ["general"])
        
        # Score models based on capabilities and priority
        scored_models = []
        for name, model in self.models.items():
            if not model.enabled:
                continue
            
            # Check if model is available
            if not await self.check_model_availability(name):
                continue
            
            # Calculate capability score
            capability_score = sum(1 for cap in required_capabilities if cap in model.capabilities)
            total_score = capability_score * model.priority
            
            scored_models.append((name, total_score))
        
        # Return highest scoring model
        if scored_models:
            scored_models.sort(key=lambda x: x[1], reverse=True)
            return scored_models[0][0]
        
        return None

# Global model manager instance
model_manager = ModelManager()

async def main():
    """Test model manager"""
    logging.basicConfig(level=logging.INFO)
    
    # Check model availability
    for model_name in model_manager.models:
        available = await model_manager.check_model_availability(model_name)
        print(f"{model_name}: {'Available' if available else 'Not Available'}")
    
    # Test response generation
    response = await model_manager.generate_response("Hello, how are you?")
    print(f"Response: {response.content}")
    print(f"Model used: {response.model_used}")
    print(f"Success: {response.success}")

if __name__ == "__main__":
    asyncio.run(main())