#!/usr/bin/env python3
"""
WADE Settings Manager - Complete configuration system
Handles all WADE settings including models, execution, autonomy, security, etc.
"""

import json
import yaml
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime

@dataclass
class ModelSettings:
    default_reasoning_model: str = "phind-codellama"
    model_router_strategy: str = "auto"  # manual, auto, scored
    token_limit_per_chain: int = 4096
    prompt_style_bias: str = "logic"  # logic, creative, stealthy
    temperature: float = 0.1
    fallback_models: List[str] = field(default_factory=lambda: ["deepseek-coder", "wizardlm"])

@dataclass
class ExecutionSettings:
    execution_mode: str = "simulation"  # simulation, live
    confirm_before_live: bool = True
    max_chains_in_sandbox: int = 10
    allow_payload_write: bool = False
    require_root_for_live: bool = True
    execution_timeout: int = 300  # seconds

@dataclass
class AutonomySettings:
    autonomy_mode: bool = False
    execution_frequency: int = 60  # minutes
    success_threshold: float = 0.8
    enable_mutation: bool = True
    auto_tune_prompt_chains: bool = True
    max_autonomous_runs: int = 5
    learning_rate: float = 0.1

@dataclass
class NetworkSettings:
    default_interface: str = "eth0"
    enable_tor_routing: bool = False
    use_proxychains: bool = False
    search_sources: List[str] = field(default_factory=lambda: ["google", "github"])
    request_delay: float = 1.0  # seconds
    max_concurrent_requests: int = 5
    user_agent: str = "WADE/1.0"

@dataclass
class MemorySettings:
    memory_path: str = "/workspace/wade_env/memory"
    retain_history_days: int = 30
    auto_cleanup: bool = True
    score_display_mode: str = "normalized"  # raw, normalized, confidence
    max_memory_size_mb: int = 1000
    backup_frequency: int = 24  # hours

@dataclass
class SecuritySettings:
    kill_switch_keybind: str = "Ctrl+Alt+Esc"
    require_root_for_live: bool = True
    execution_audit_logging: bool = True
    password_required_for_live: bool = False
    live_mode_password: str = ""
    panic_button_enabled: bool = True
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [".py", ".sh", ".txt", ".json", ".yaml"])

@dataclass
class SystemSettings:
    default_startup_agents: List[str] = field(default_factory=lambda: ["Planner", "Executor", "Memory"])
    enable_cli_hooks: bool = True
    run_at_boot: bool = False
    whisper_voice_input: bool = False
    local_model_path: str = "/workspace/wade_env/models"
    workspace_path: str = "/workspace/wade_env"
    log_level: str = "INFO"

@dataclass
class UISettings:
    theme: str = "dark"  # dark, light, auto
    show_persistent_memory: bool = True
    show_chat_history: bool = True
    show_project_files: bool = True
    auto_save_interval: int = 30  # seconds
    max_chat_history: int = 1000
    terminal_font_size: int = 14

@dataclass
class WADESettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    autonomy: AutonomySettings = field(default_factory=AutonomySettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    system: SystemSettings = field(default_factory=SystemSettings)
    ui: UISettings = field(default_factory=UISettings)
    version: str = "1.0.0"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

class SettingsManager:
    def __init__(self, config_path: str = "/workspace/wade_env/wade_settings.yaml"):
        self.config_path = Path(config_path)
        self.settings = WADESettings()
        self.profiles_dir = Path("/workspace/wade_env/profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.load_settings()
    
    def load_settings(self):
        """Load settings from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Update settings with loaded data
                if data:
                    self._update_settings_from_dict(data)
                
                logging.info("Settings loaded successfully")
                
            except Exception as e:
                logging.error(f"Error loading settings: {e}")
                self._create_default_settings()
        else:
            self._create_default_settings()
    
    def _update_settings_from_dict(self, data: Dict[str, Any]):
        """Update settings from dictionary"""
        try:
            # Update each section
            if 'model' in data:
                self.settings.model = ModelSettings(**data['model'])
            if 'execution' in data:
                self.settings.execution = ExecutionSettings(**data['execution'])
            if 'autonomy' in data:
                self.settings.autonomy = AutonomySettings(**data['autonomy'])
            if 'network' in data:
                self.settings.network = NetworkSettings(**data['network'])
            if 'memory' in data:
                self.settings.memory = MemorySettings(**data['memory'])
            if 'security' in data:
                self.settings.security = SecuritySettings(**data['security'])
            if 'system' in data:
                self.settings.system = SystemSettings(**data['system'])
            if 'ui' in data:
                self.settings.ui = UISettings(**data['ui'])
            
            # Update metadata
            self.settings.version = data.get('version', self.settings.version)
            self.settings.last_updated = datetime.now().isoformat()
            
        except Exception as e:
            logging.error(f"Error updating settings: {e}")
    
    def _create_default_settings(self):
        """Create default settings"""
        self.settings = WADESettings()
        self.save_settings()
        logging.info("Created default settings")
    
    def save_settings(self):
        """Save settings to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            settings_dict = asdict(self.settings)
            settings_dict['last_updated'] = datetime.now().isoformat()
            
            with open(self.config_path, 'w') as f:
                yaml.dump(settings_dict, f, default_flow_style=False, indent=2)
            
            logging.info("Settings saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
    
    def get_settings_dict(self) -> Dict[str, Any]:
        """Get settings as dictionary"""
        return asdict(self.settings)
    
    def update_settings(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update specific settings section"""
        try:
            if section == "model":
                for key, value in updates.items():
                    if hasattr(self.settings.model, key):
                        setattr(self.settings.model, key, value)
            elif section == "execution":
                for key, value in updates.items():
                    if hasattr(self.settings.execution, key):
                        setattr(self.settings.execution, key, value)
            elif section == "autonomy":
                for key, value in updates.items():
                    if hasattr(self.settings.autonomy, key):
                        setattr(self.settings.autonomy, key, value)
            elif section == "network":
                for key, value in updates.items():
                    if hasattr(self.settings.network, key):
                        setattr(self.settings.network, key, value)
            elif section == "memory":
                for key, value in updates.items():
                    if hasattr(self.settings.memory, key):
                        setattr(self.settings.memory, key, value)
            elif section == "security":
                for key, value in updates.items():
                    if hasattr(self.settings.security, key):
                        setattr(self.settings.security, key, value)
            elif section == "system":
                for key, value in updates.items():
                    if hasattr(self.settings.system, key):
                        setattr(self.settings.system, key, value)
            elif section == "ui":
                for key, value in updates.items():
                    if hasattr(self.settings.ui, key):
                        setattr(self.settings.ui, key, value)
            else:
                return False
            
            self.save_settings()
            return True
            
        except Exception as e:
            logging.error(f"Error updating settings section {section}: {e}")
            return False
    
    def save_profile(self, profile_name: str) -> bool:
        """Save current settings as a profile"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            
            profile_data = asdict(self.settings)
            profile_data['profile_name'] = profile_name
            profile_data['created_at'] = datetime.now().isoformat()
            
            with open(profile_path, 'w') as f:
                yaml.dump(profile_data, f, default_flow_style=False, indent=2)
            
            logging.info(f"Profile '{profile_name}' saved")
            return True
            
        except Exception as e:
            logging.error(f"Error saving profile {profile_name}: {e}")
            return False
    
    def load_profile(self, profile_name: str) -> bool:
        """Load settings from a profile"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            
            if not profile_path.exists():
                logging.error(f"Profile '{profile_name}' not found")
                return False
            
            with open(profile_path, 'r') as f:
                profile_data = yaml.safe_load(f)
            
            # Remove profile metadata before loading
            profile_data.pop('profile_name', None)
            profile_data.pop('created_at', None)
            
            self._update_settings_from_dict(profile_data)
            self.save_settings()
            
            logging.info(f"Profile '{profile_name}' loaded")
            return True
            
        except Exception as e:
            logging.error(f"Error loading profile {profile_name}: {e}")
            return False
    
    def list_profiles(self) -> List[Dict[str, str]]:
        """List available profiles"""
        profiles = []
        
        for profile_file in self.profiles_dir.glob("*.yaml"):
            try:
                with open(profile_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                profiles.append({
                    "name": profile_file.stem,
                    "display_name": data.get('profile_name', profile_file.stem),
                    "created_at": data.get('created_at', 'Unknown'),
                    "version": data.get('version', 'Unknown')
                })
                
            except Exception as e:
                logging.error(f"Error reading profile {profile_file}: {e}")
        
        return sorted(profiles, key=lambda x: x['created_at'], reverse=True)
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            
            if profile_path.exists():
                profile_path.unlink()
                logging.info(f"Profile '{profile_name}' deleted")
                return True
            else:
                logging.error(f"Profile '{profile_name}' not found")
                return False
                
        except Exception as e:
            logging.error(f"Error deleting profile {profile_name}: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.settings = WADESettings()
        self.save_settings()
        logging.info("Settings reset to defaults")
    
    def validate_settings(self) -> List[str]:
        """Validate current settings and return any issues"""
        issues = []
        
        # Validate model settings
        if self.settings.model.token_limit_per_chain < 100:
            issues.append("Token limit per chain is too low (minimum 100)")
        
        # Validate execution settings
        if self.settings.execution.execution_timeout < 10:
            issues.append("Execution timeout is too low (minimum 10 seconds)")
        
        # Validate autonomy settings
        if self.settings.autonomy.success_threshold < 0 or self.settings.autonomy.success_threshold > 1:
            issues.append("Success threshold must be between 0 and 1")
        
        # Validate network settings
        if self.settings.network.request_delay < 0:
            issues.append("Request delay cannot be negative")
        
        # Validate memory settings
        if self.settings.memory.retain_history_days < 1:
            issues.append("History retention must be at least 1 day")
        
        # Validate security settings
        if self.settings.security.max_file_size_mb < 1:
            issues.append("Maximum file size must be at least 1 MB")
        
        # Validate paths
        memory_path = Path(self.settings.memory.memory_path)
        if not memory_path.parent.exists():
            issues.append(f"Memory path parent directory does not exist: {memory_path.parent}")
        
        workspace_path = Path(self.settings.system.workspace_path)
        if not workspace_path.exists():
            issues.append(f"Workspace path does not exist: {workspace_path}")
        
        return issues
    
    def get_setting_options(self) -> Dict[str, Dict[str, List[str]]]:
        """Get available options for dropdown/select settings"""
        return {
            "model": {
                "model_router_strategy": ["manual", "auto", "scored"],
                "prompt_style_bias": ["logic", "creative", "stealthy"]
            },
            "execution": {
                "execution_mode": ["simulation", "live"]
            },
            "network": {
                "default_interface": ["eth0", "wlan0", "lo", "docker0"],
                "search_sources": ["google", "github", "shodan", "dnsdb", "virustotal"]
            },
            "memory": {
                "score_display_mode": ["raw", "normalized", "confidence"]
            },
            "system": {
                "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
                "default_startup_agents": ["Planner", "Executor", "Memory", "Coder", "Tester", "Analyzer"]
            },
            "ui": {
                "theme": ["dark", "light", "auto"]
            }
        }

# Global settings manager instance
settings_manager = SettingsManager()

def main():
    """Test settings manager"""
    logging.basicConfig(level=logging.INFO)
    
    # Print current settings
    print("Current Settings:")
    print(json.dumps(settings_manager.get_settings_dict(), indent=2, default=str))
    
    # Test profile operations
    settings_manager.save_profile("test_profile")
    profiles = settings_manager.list_profiles()
    print(f"\nAvailable profiles: {[p['name'] for p in profiles]}")
    
    # Test validation
    issues = settings_manager.validate_settings()
    if issues:
        print(f"\nValidation issues: {issues}")
    else:
        print("\nSettings validation passed")

if __name__ == "__main__":
    main()