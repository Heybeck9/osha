#!/usr/bin/env python3
"""
WADE VS Code Service - Real VS Code server integration
Manages code-server instances and provides API access
"""

import asyncio
import subprocess
import os
import signal
import json
import requests
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

class VSCodeService:
    def __init__(self, workspace_path: str = "/workspace/wade_env"):
        self.workspace_path = Path(workspace_path)
        self.port = 12001
        self.host = "0.0.0.0"
        self.process: Optional[subprocess.Popen] = None
        self.config_dir = Path.home() / ".config" / "code-server"
        self.running = False
        
    async def install_code_server(self):
        """Install code-server if not already installed"""
        try:
            # Check if code-server is already installed
            result = subprocess.run(["which", "code-server"], capture_output=True)
            if result.returncode == 0:
                logging.info("code-server already installed")
                return True
            
            logging.info("Installing code-server...")
            
            # Download and install code-server
            install_script = """
            curl -fsSL https://code-server.dev/install.sh | sh
            """
            
            process = await asyncio.create_subprocess_shell(
                install_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logging.info("code-server installed successfully")
                return True
            else:
                logging.error(f"Failed to install code-server: {stderr.decode()}")
                return False
                
        except Exception as e:
            logging.error(f"Error installing code-server: {e}")
            return False
    
    def create_config(self):
        """Create VS Code server configuration"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "bind-addr": f"{self.host}:{self.port}",
            "auth": "none",
            "cert": False,
            "disable-telemetry": True,
            "disable-update-check": True
        }
        
        config_file = self.config_dir / "config.yaml"
        
        with open(config_file, 'w') as f:
            for key, value in config.items():
                if isinstance(value, bool):
                    f.write(f"{key}: {str(value).lower()}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logging.info(f"VS Code config created at {config_file}")
    
    async def start_server(self):
        """Start VS Code server"""
        if self.running:
            logging.info("VS Code server already running")
            return True
        
        try:
            # Ensure code-server is installed
            if not await self.install_code_server():
                return False
            
            # Create configuration
            self.create_config()
            
            # Ensure workspace directory exists
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Start code-server
            cmd = [
                "code-server",
                "--bind-addr", f"{self.host}:{self.port}",
                "--auth", "none",
                "--disable-telemetry",
                "--disable-update-check",
                str(self.workspace_path)
            ]
            
            logging.info(f"Starting VS Code server: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for server to start
            await self._wait_for_server()
            
            if self.is_running():
                self.running = True
                logging.info(f"VS Code server started on http://{self.host}:{self.port}")
                return True
            else:
                logging.error("VS Code server failed to start")
                return False
                
        except Exception as e:
            logging.error(f"Error starting VS Code server: {e}")
            return False
    
    async def _wait_for_server(self, timeout: int = 30):
        """Wait for VS Code server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_running():
                return True
            await asyncio.sleep(1)
        
        return False
    
    def is_running(self) -> bool:
        """Check if VS Code server is running"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def stop_server(self):
        """Stop VS Code server"""
        if self.process:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for process to terminate
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                
                self.process = None
                self.running = False
                logging.info("VS Code server stopped")
                
            except Exception as e:
                logging.error(f"Error stopping VS Code server: {e}")
    
    async def restart_server(self):
        """Restart VS Code server"""
        await self.stop_server()
        await asyncio.sleep(2)
        return await self.start_server()
    
    def get_server_url(self) -> str:
        """Get VS Code server URL"""
        return f"http://{self.host}:{self.port}"
    
    async def create_file(self, file_path: str, content: str = "") -> bool:
        """Create a file in the workspace"""
        try:
            full_path = self.workspace_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
            
            logging.info(f"Created file: {full_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating file {file_path}: {e}")
            return False
    
    async def read_file(self, file_path: str) -> Optional[str]:
        """Read a file from the workspace"""
        try:
            full_path = self.workspace_path / file_path
            
            if full_path.exists():
                with open(full_path, 'r') as f:
                    return f.read()
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None
    
    async def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file"""
        try:
            full_path = self.workspace_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
            
            logging.info(f"Updated file: {full_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error writing file {file_path}: {e}")
            return False
    
    async def list_files(self, directory: str = "") -> List[Dict[str, str]]:
        """List files in workspace directory"""
        try:
            target_dir = self.workspace_path / directory if directory else self.workspace_path
            
            if not target_dir.exists():
                return []
            
            files = []
            for item in target_dir.iterdir():
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(self.workspace_path)),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0
                })
            
            return sorted(files, key=lambda x: (x["type"] == "file", x["name"]))
            
        except Exception as e:
            logging.error(f"Error listing files in {directory}: {e}")
            return []
    
    async def install_extension(self, extension_id: str) -> bool:
        """Install VS Code extension"""
        try:
            cmd = ["code-server", "--install-extension", extension_id]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logging.info(f"Extension {extension_id} installed successfully")
                return True
            else:
                logging.error(f"Failed to install extension {extension_id}: {stderr.decode()}")
                return False
                
        except Exception as e:
            logging.error(f"Error installing extension {extension_id}: {e}")
            return False
    
    async def setup_workspace(self):
        """Set up the workspace with essential files and extensions"""
        try:
            # Create essential directories
            directories = [
                "src",
                "tests", 
                "docs",
                "scripts",
                "config"
            ]
            
            for directory in directories:
                (self.workspace_path / directory).mkdir(exist_ok=True)
            
            # Create essential files
            files = {
                "README.md": "# WADE Workspace\n\nAutonomous development environment workspace.\n",
                "requirements.txt": "# Python dependencies\nfastapi\nuvicorn\nrequests\n",
                ".gitignore": "__pycache__/\n*.pyc\n.env\nvenv/\n.vscode/\n",
                "src/__init__.py": "# WADE source code\n",
                "tests/__init__.py": "# WADE tests\n"
            }
            
            for file_path, content in files.items():
                await self.create_file(file_path, content)
            
            # Install essential extensions
            extensions = [
                "ms-python.python",
                "ms-python.pylint",
                "ms-python.black-formatter",
                "ms-vscode.vscode-json",
                "redhat.vscode-yaml"
            ]
            
            for extension in extensions:
                await self.install_extension(extension)
            
            logging.info("Workspace setup completed")
            return True
            
        except Exception as e:
            logging.error(f"Error setting up workspace: {e}")
            return False

# Global VS Code service instance
vscode_service = VSCodeService()

async def main():
    """Test VS Code service"""
    logging.basicConfig(level=logging.INFO)
    
    # Start VS Code server
    if await vscode_service.start_server():
        print(f"VS Code server running at: {vscode_service.get_server_url()}")
        
        # Set up workspace
        await vscode_service.setup_workspace()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await vscode_service.stop_server()
    else:
        print("Failed to start VS Code server")

if __name__ == "__main__":
    asyncio.run(main())