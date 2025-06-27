"""
Error Handler for WADE autonomous development environment.
Provides centralized error handling, logging, and notification.
"""

import asyncio
import datetime
import json
import logging
import os
import traceback
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable, Type, Union

# Import WADE components
try:
    from websocket_manager import websocket_manager
except ImportError:
    # For standalone testing
    from wade_env.websocket_manager import websocket_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("error_handler")

class ErrorSeverity(Enum):
    """Enum for error severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ErrorCategory(Enum):
    """Enum for error categories."""
    SYSTEM = "system"
    SECURITY = "security"
    MODEL = "model"
    DATABASE = "database"
    NETWORK = "network"
    USER = "user"
    AGENT = "agent"
    UNKNOWN = "unknown"

class ErrorHandler:
    """
    Centralized error handling system.
    Provides error logging, notification, and recovery mechanisms.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_log: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.error_callbacks: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        self.recovery_handlers: Dict[Type[Exception], Callable[[Exception], Awaitable[bool]]] = {}
        self.max_log_size = 1000
        self.notification_threshold = 3  # Notify after 3 occurrences of the same error
        self.error_categories = {
            "PermissionError": ErrorCategory.SECURITY,
            "ConnectionError": ErrorCategory.NETWORK,
            "TimeoutError": ErrorCategory.NETWORK,
            "ValueError": ErrorCategory.USER,
            "KeyError": ErrorCategory.SYSTEM,
            "IndexError": ErrorCategory.SYSTEM,
            "TypeError": ErrorCategory.SYSTEM,
            "RuntimeError": ErrorCategory.SYSTEM,
            "AgentError": ErrorCategory.AGENT,
            "ModelError": ErrorCategory.MODEL,
            "DatabaseError": ErrorCategory.DATABASE
        }
    
    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle an error and return an error ID.
        This is the main entry point for error handling.
        """
        error_id = str(uuid.uuid4())
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Determine error category and severity
        category = self._get_error_category(error)
        severity = self._get_error_severity(error)
        
        # Create error record
        error_record = {
            "id": error_id,
            "type": error_type,
            "message": error_message,
            "traceback": error_traceback,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "category": category.value,
            "severity": severity.value,
            "context": context or {}
        }
        
        # Log the error
        self._log_error(error_record)
        
        # Update error counts
        error_key = f"{error_type}:{error_message}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Check if we should notify
        if self.error_counts[error_key] >= self.notification_threshold:
            await self._notify_error(error_record)
            # Reset count after notification
            self.error_counts[error_key] = 0
        
        # Try to recover
        await self._attempt_recovery(error, error_record)
        
        # Trigger callbacks
        await self._trigger_callbacks(error_record)
        
        return error_id
    
    def _get_error_category(self, error: Exception) -> ErrorCategory:
        """Determine the category of an error."""
        error_type = type(error).__name__
        return self.error_categories.get(error_type, ErrorCategory.UNKNOWN)
    
    def _get_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine the severity of an error."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL
        
        # Security errors
        if error_type in ["PermissionError", "AuthenticationError"]:
            return ErrorSeverity.CRITICAL
        
        # Network errors
        if error_type in ["ConnectionError", "TimeoutError"]:
            return ErrorSeverity.ERROR
        
        # User errors
        if error_type in ["ValueError", "KeyError", "IndexError"]:
            return ErrorSeverity.WARNING
        
        # Default to ERROR
        return ErrorSeverity.ERROR
    
    def _log_error(self, error_record: Dict[str, Any]) -> None:
        """Log an error to the internal log and system logger."""
        # Add to internal log
        self.error_log.append(error_record)
        
        # Limit log size
        if len(self.error_log) > self.max_log_size:
            self.error_log = self.error_log[-self.max_log_size:]
        
        # Log to system logger
        severity = ErrorSeverity(error_record["severity"])
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"[{error_record['id']}] {error_record['type']}: {error_record['message']}")
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"[{error_record['id']}] {error_record['type']}: {error_record['message']}")
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"[{error_record['id']}] {error_record['type']}: {error_record['message']}")
        elif severity == ErrorSeverity.INFO:
            logger.info(f"[{error_record['id']}] {error_record['type']}: {error_record['message']}")
        else:
            logger.debug(f"[{error_record['id']}] {error_record['type']}: {error_record['message']}")
        
        # Log to file
        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "errors.log")
            with open(log_file, "a") as f:
                f.write(json.dumps(error_record) + "\n")
        except Exception as e:
            logger.error(f"Error writing to error log file: {e}")
    
    async def _notify_error(self, error_record: Dict[str, Any]) -> None:
        """Notify about an error via WebSocket."""
        try:
            # Prepare notification data
            notification = {
                "error_id": error_record["id"],
                "type": error_record["type"],
                "message": error_record["message"],
                "category": error_record["category"],
                "severity": error_record["severity"],
                "timestamp": error_record["timestamp"]
            }
            
            # Broadcast via WebSocket
            await websocket_manager.broadcast("error", notification)
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")
    
    async def _attempt_recovery(self, error: Exception, error_record: Dict[str, Any]) -> bool:
        """Attempt to recover from an error."""
        error_type = type(error)
        
        # Check if we have a recovery handler for this error type
        for exception_type, handler in self.recovery_handlers.items():
            if isinstance(error, exception_type):
                try:
                    recovered = await handler(error)
                    
                    # Update error record with recovery status
                    error_record["recovered"] = recovered
                    
                    if recovered:
                        logger.info(f"Successfully recovered from {error_type.__name__}")
                    
                    return recovered
                except Exception as recovery_error:
                    logger.error(f"Error in recovery handler: {recovery_error}")
                    return False
        
        # No recovery handler found
        error_record["recovered"] = False
        return False
    
    async def _trigger_callbacks(self, error_record: Dict[str, Any]) -> None:
        """Trigger callbacks for an error."""
        error_type = error_record["type"]
        
        # Trigger callbacks for this error type
        if error_type in self.error_callbacks:
            for callback in self.error_callbacks[error_type]:
                try:
                    await callback(error_record)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
        
        # Trigger callbacks for all errors
        if "*" in self.error_callbacks:
            for callback in self.error_callbacks["*"]:
                try:
                    await callback(error_record)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
    
    def register_callback(self, error_type: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a callback for a specific error type."""
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        
        self.error_callbacks[error_type].append(callback)
    
    def register_recovery_handler(self, exception_type: Type[Exception], 
                                 handler: Callable[[Exception], Awaitable[bool]]) -> None:
        """Register a recovery handler for a specific exception type."""
        self.recovery_handlers[exception_type] = handler
    
    def get_error_log(self, limit: int = 100, 
                     severity: Optional[Union[ErrorSeverity, int]] = None,
                     category: Optional[Union[ErrorCategory, str]] = None) -> List[Dict[str, Any]]:
        """Get the error log, optionally filtered by severity and category."""
        filtered_log = self.error_log
        
        # Filter by severity
        if severity is not None:
            if isinstance(severity, ErrorSeverity):
                severity_value = severity.value
            else:
                severity_value = severity
            
            filtered_log = [
                error for error in filtered_log
                if error["severity"] >= severity_value
            ]
        
        # Filter by category
        if category is not None:
            if isinstance(category, ErrorCategory):
                category_value = category.value
            else:
                category_value = category
            
            filtered_log = [
                error for error in filtered_log
                if error["category"] == category_value
            ]
        
        # Sort by timestamp (newest first)
        sorted_log = sorted(
            filtered_log,
            key=lambda error: error["timestamp"],
            reverse=True
        )
        
        # Limit results
        return sorted_log[:limit]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_log:
            return {
                "total_errors": 0,
                "by_category": {},
                "by_severity": {},
                "recent_errors": []
            }
        
        # Count errors by category and severity
        categories: Dict[str, int] = {}
        severities: Dict[int, int] = {}
        
        for error in self.error_log:
            # Count by category
            category = error["category"]
            categories[category] = categories.get(category, 0) + 1
            
            # Count by severity
            severity = error["severity"]
            severities[severity] = severities.get(severity, 0) + 1
        
        # Get recent errors
        recent_errors = self.get_error_log(limit=5)
        
        return {
            "total_errors": len(self.error_log),
            "by_category": categories,
            "by_severity": severities,
            "recent_errors": [
                {
                    "id": error["id"],
                    "type": error["type"],
                    "message": error["message"],
                    "timestamp": error["timestamp"]
                }
                for error in recent_errors
            ]
        }
    
    def clear_error_log(self) -> None:
        """Clear the error log."""
        self.error_log = []
        self.error_counts = {}

# Create singleton instance
error_handler = ErrorHandler()