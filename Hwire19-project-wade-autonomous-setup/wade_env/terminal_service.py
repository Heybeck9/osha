#!/usr/bin/env python3
"""
WADE Terminal Service - Real terminal integration with WebSocket
Provides actual shell access through web interface
"""

import asyncio
import json
import pty
import os
import signal
import subprocess
import threading
from typing import Dict, List, Optional
import websockets
import logging

class TerminalSession:
    def __init__(self, session_id: str, shell: str = "/bin/bash"):
        self.session_id = session_id
        self.shell = shell
        self.process = None
        self.master_fd = None
        self.slave_fd = None
        self.websockets: List = []
        self.running = False
        self.cwd = os.getcwd()
        
    async def start(self):
        """Start the terminal session"""
        try:
            # Create pseudo-terminal
            self.master_fd, self.slave_fd = pty.openpty()
            
            # Start shell process
            self.process = subprocess.Popen(
                [self.shell],
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                cwd=self.cwd,
                env=os.environ.copy(),
                preexec_fn=os.setsid
            )
            
            self.running = True
            
            # Start reading from terminal
            asyncio.create_task(self._read_output())
            
            logging.info(f"Terminal session {self.session_id} started")
            
        except Exception as e:
            logging.error(f"Failed to start terminal session: {e}")
            raise
    
    async def _read_output(self):
        """Read output from terminal and broadcast to websockets"""
        loop = asyncio.get_event_loop()
        
        def read_from_fd():
            try:
                return os.read(self.master_fd, 1024)
            except OSError:
                return b''
        
        while self.running and self.process and self.process.poll() is None:
            try:
                # Read from terminal in a thread to avoid blocking
                data = await loop.run_in_executor(None, read_from_fd)
                
                if data:
                    output = data.decode('utf-8', errors='replace')
                    await self._broadcast_output(output)
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"Error reading terminal output: {e}")
                break
        
        self.running = False
    
    async def _broadcast_output(self, output: str):
        """Broadcast output to all connected websockets"""
        if not self.websockets:
            return
        
        message = {
            "type": "terminal_output",
            "session_id": self.session_id,
            "data": output
        }
        
        # Remove closed websockets
        active_websockets = []
        for ws in self.websockets:
            try:
                await ws.send(json.dumps(message))
                active_websockets.append(ws)
            except:
                pass
        
        self.websockets = active_websockets
    
    async def write_input(self, data: str):
        """Write input to terminal"""
        if self.master_fd and self.running:
            try:
                os.write(self.master_fd, data.encode('utf-8'))
            except Exception as e:
                logging.error(f"Error writing to terminal: {e}")
    
    def add_websocket(self, websocket):
        """Add websocket connection"""
        self.websockets.append(websocket)
    
    def remove_websocket(self, websocket):
        """Remove websocket connection"""
        if websocket in self.websockets:
            self.websockets.remove(websocket)
    
    async def resize(self, rows: int, cols: int):
        """Resize terminal"""
        if self.master_fd:
            try:
                import fcntl
                import termios
                import struct
                
                # Set terminal size
                size = struct.pack('HHHH', rows, cols, 0, 0)
                fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, size)
                
            except Exception as e:
                logging.error(f"Error resizing terminal: {e}")
    
    async def stop(self):
        """Stop terminal session"""
        self.running = False
        
        if self.process:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for process to terminate
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    
            except Exception as e:
                logging.error(f"Error stopping terminal process: {e}")
        
        # Close file descriptors
        if self.master_fd:
            try:
                os.close(self.master_fd)
            except:
                pass
        
        if self.slave_fd:
            try:
                os.close(self.slave_fd)
            except:
                pass
        
        logging.info(f"Terminal session {self.session_id} stopped")

class TerminalService:
    def __init__(self):
        self.sessions: Dict[str, TerminalSession] = {}
        self.websocket_sessions: Dict = {}  # websocket -> session_id mapping
    
    async def create_session(self, session_id: str, shell: str = "/bin/bash") -> TerminalSession:
        """Create a new terminal session"""
        if session_id in self.sessions:
            await self.sessions[session_id].stop()
        
        session = TerminalSession(session_id, shell)
        await session.start()
        self.sessions[session_id] = session
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[TerminalSession]:
        """Get existing terminal session"""
        return self.sessions.get(session_id)
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connection for terminal"""
        session_id = None
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "terminal_connect":
                        session_id = data.get("session_id", "default")
                        
                        # Get or create session
                        session = await self.get_session(session_id)
                        if not session:
                            session = await self.create_session(session_id)
                        
                        # Add websocket to session
                        session.add_websocket(websocket)
                        self.websocket_sessions[websocket] = session_id
                        
                        # Send connection confirmation
                        await websocket.send(json.dumps({
                            "type": "terminal_connected",
                            "session_id": session_id
                        }))
                    
                    elif msg_type == "terminal_input":
                        if session_id:
                            session = await self.get_session(session_id)
                            if session:
                                await session.write_input(data.get("data", ""))
                    
                    elif msg_type == "terminal_resize":
                        if session_id:
                            session = await self.get_session(session_id)
                            if session:
                                rows = data.get("rows", 24)
                                cols = data.get("cols", 80)
                                await session.resize(rows, cols)
                    
                    elif msg_type == "terminal_command":
                        # Execute specific command
                        if session_id:
                            session = await self.get_session(session_id)
                            if session:
                                command = data.get("command", "")
                                await session.write_input(command + "\n")
                
                except json.JSONDecodeError:
                    logging.error("Invalid JSON received")
                except Exception as e:
                    logging.error(f"Error handling websocket message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
        finally:
            # Clean up
            if websocket in self.websocket_sessions:
                session_id = self.websocket_sessions[websocket]
                session = await self.get_session(session_id)
                if session:
                    session.remove_websocket(websocket)
                del self.websocket_sessions[websocket]
    
    async def execute_command(self, session_id: str, command: str) -> str:
        """Execute a command and return output"""
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
        
        # Send command
        await session.write_input(command + "\n")
        
        # Note: For real-time output, use WebSocket connection
        # This method is for simple command execution
        return f"Command '{command}' sent to terminal session {session_id}"
    
    async def stop_session(self, session_id: str):
        """Stop a terminal session"""
        if session_id in self.sessions:
            await self.sessions[session_id].stop()
            del self.sessions[session_id]
    
    async def stop_all_sessions(self):
        """Stop all terminal sessions"""
        for session in self.sessions.values():
            await session.stop()
        self.sessions.clear()

# Global terminal service instance
terminal_service = TerminalService()

async def start_terminal_server(host="localhost", port=12003):
    """Start the terminal WebSocket server"""
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting terminal server on {host}:{port}")
    
    async with websockets.serve(terminal_service.handle_websocket, host, port):
        logging.info("Terminal server started")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Run terminal server
    asyncio.run(start_terminal_server())