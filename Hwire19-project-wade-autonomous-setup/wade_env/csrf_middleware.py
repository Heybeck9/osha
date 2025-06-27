"""
CSRF Middleware for WADE autonomous development environment.
Provides protection against Cross-Site Request Forgery attacks.
"""

import secrets
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
import hashlib
import hmac
import json
import os
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csrf_middleware")

class CSRFMiddleware:
    """
    CSRF protection middleware for FastAPI applications.
    Implements token-based CSRF protection.
    """
    
    def __init__(self):
        """Initialize the CSRF middleware."""
        self.secret_key = self._get_or_create_secret()
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.token_expiry = 3600  # 1 hour
        self.exempt_paths = [
            "/static/",
            "/api/auth/login",
            "/api/auth/token"
        ]
        self.exempt_methods = ["GET", "HEAD", "OPTIONS"]
    
    def _get_or_create_secret(self) -> bytes:
        """Get or create the CSRF secret."""
        secret_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "csrf_secret.key"
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(secret_file), exist_ok=True)
        
        if os.path.exists(secret_file):
            # Load existing secret
            with open(secret_file, "rb") as f:
                return f.read()
        else:
            # Generate new secret
            secret = secrets.token_bytes(32)
            with open(secret_file, "wb") as f:
                f.write(secret)
            return secret
    
    def generate_token(self, user_id: str = "anonymous") -> str:
        """
        Generate a CSRF token.
        Returns the token string.
        """
        # Generate token
        token = secrets.token_hex(16)
        
        # Store token
        self.tokens[token] = {
            "user_id": user_id,
            "created_at": time.time(),
            "expires_at": time.time() + self.token_expiry
        }
        
        return token
    
    def validate_token(self, token: str) -> bool:
        """
        Validate a CSRF token.
        Returns True if the token is valid, False otherwise.
        """
        if token not in self.tokens:
            return False
        
        token_data = self.tokens[token]
        
        # Check if token has expired
        if token_data["expires_at"] < time.time():
            # Remove expired token
            del self.tokens[token]
            return False
        
        return True
    
    def is_path_exempt(self, path: str) -> bool:
        """
        Check if a path is exempt from CSRF protection.
        Returns True if the path is exempt, False otherwise.
        """
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        
        return False
    
    def is_method_exempt(self, method: str) -> bool:
        """
        Check if a method is exempt from CSRF protection.
        Returns True if the method is exempt, False otherwise.
        """
        return method.upper() in self.exempt_methods
    
    def clean_expired_tokens(self) -> None:
        """Clean expired tokens."""
        current_time = time.time()
        expired_tokens = [
            token for token, data in self.tokens.items()
            if data["expires_at"] < current_time
        ]
        
        for token in expired_tokens:
            del self.tokens[token]
    
    def add_exempt_path(self, path: str) -> None:
        """Add a path to the exempt paths list."""
        if path not in self.exempt_paths:
            self.exempt_paths.append(path)
    
    def add_exempt_method(self, method: str) -> None:
        """Add a method to the exempt methods list."""
        method = method.upper()
        if method not in self.exempt_methods:
            self.exempt_methods.append(method)
    
    def get_token_count(self) -> int:
        """Get the number of active tokens."""
        return len(self.tokens)

# Create singleton instance
csrf_middleware = CSRFMiddleware()