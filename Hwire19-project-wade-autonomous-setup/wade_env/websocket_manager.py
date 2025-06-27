"""
WebSocket Manager for WADE autonomous development environment.
Provides real-time updates and notifications to connected clients.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Callable, Awaitable
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_manager")

class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts messages to connected clients.
    Provides real-time updates for system events, agent status, and more.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, Any] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        self.broadcast_tasks: Dict[str, asyncio.Task] = {}
        self.broadcast_intervals: Dict[str, float] = {
            "agent_status": 1.0,  # Every 1 second
            "performance_metrics": 5.0,  # Every 5 seconds
            "evolution_status": 10.0  # Every 10 seconds
        }
    
    async def connect(self, websocket: Any) -> str:
        """
        Register a new WebSocket connection.
        Returns a connection ID that can be used to identify this connection.
        """
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "connected_at": time.time(),
            "client_info": self._get_client_info(websocket),
            "last_activity": time.time()
        }
        
        logger.info(f"New WebSocket connection: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    def _get_client_info(self, websocket: Any) -> Dict[str, Any]:
        """Extract client information from the WebSocket connection."""
        try:
            # This will depend on the WebSocket implementation
            # For FastAPI WebSockets:
            headers = getattr(websocket, "headers", {})
            client_host = getattr(websocket, "client", {}).get("host", "unknown")
            
            return {
                "user_agent": headers.get("user-agent", "unknown"),
                "ip_address": client_host,
                "origin": headers.get("origin", "unknown")
            }
        except Exception as e:
            logger.error(f"Error getting client info: {e}")
            return {"error": "Could not retrieve client info"}
    
    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        json_message = json.dumps(message)
        
        # Trigger any registered event handlers
        await self._trigger_event_handlers(event_type, data)
        
        # Send to all connected clients
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json_message)
                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = time.time()
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_to(self, connection_ids: List[str], event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a message to specific clients."""
        if not connection_ids:
            return
        
        message = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        json_message = json.dumps(message)
        
        # Send to specified clients
        disconnected = []
        for connection_id in connection_ids:
            if connection_id not in self.active_connections:
                continue
                
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json_message)
                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = time.time()
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def _trigger_event_handlers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger all registered handlers for an event type."""
        if event_type not in self.event_handlers:
            return
        
        for handler in self.event_handlers[event_type]:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def start_broadcast_task(self, event_type: str, data_provider: Callable[[], Awaitable[Dict[str, Any]]]) -> None:
        """Start a periodic broadcast task for a specific event type."""
        if event_type in self.broadcast_tasks and not self.broadcast_tasks[event_type].done():
            # Task already running
            return
        
        interval = self.broadcast_intervals.get(event_type, 5.0)
        
        async def broadcast_loop():
            while True:
                try:
                    # Get data from provider
                    data = await data_provider()
                    
                    # Broadcast to all clients
                    await self.broadcast(event_type, data)
                    
                    # Wait for next interval
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    # Task was cancelled
                    break
                except Exception as e:
                    logger.error(f"Error in broadcast task for {event_type}: {e}")
                    await asyncio.sleep(interval)
        
        # Start the broadcast task
        self.broadcast_tasks[event_type] = asyncio.create_task(broadcast_loop())
        logger.info(f"Started broadcast task for {event_type}")
    
    def stop_broadcast_task(self, event_type: str) -> None:
        """Stop a periodic broadcast task."""
        if event_type in self.broadcast_tasks and not self.broadcast_tasks[event_type].done():
            self.broadcast_tasks[event_type].cancel()
            logger.info(f"Stopped broadcast task for {event_type}")
    
    def stop_all_broadcast_tasks(self) -> None:
        """Stop all periodic broadcast tasks."""
        for event_type in list(self.broadcast_tasks.keys()):
            self.stop_broadcast_task(event_type)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        current_time = time.time()
        connection_ages = [
            current_time - metadata["connected_at"]
            for metadata in self.connection_metadata.values()
        ]
        
        return {
            "active_connections": len(self.active_connections),
            "avg_connection_age": sum(connection_ages) / len(connection_ages) if connection_ages else 0,
            "oldest_connection": max(connection_ages) if connection_ages else 0,
            "newest_connection": min(connection_ages) if connection_ages else 0
        }

# Create singleton instance
websocket_manager = WebSocketManager()