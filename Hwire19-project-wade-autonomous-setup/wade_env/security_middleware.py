"""
Security Middleware for WADE autonomous development environment.
Provides request validation, rate limiting, and security headers.
"""

import time
import logging
import ipaddress
from typing import Dict, List, Any, Optional, Callable, Awaitable
import hashlib
import hmac
import json
import os
import re
from urllib.parse import urlparse

# Import WADE components
try:
    from security import request_signer, token_manager
except ImportError:
    # For standalone testing
    from wade_env.security import request_signer, token_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security_middleware")

class SecurityMiddleware:
    """
    Security middleware for FastAPI applications.
    Provides request validation, rate limiting, and security headers.
    """
    
    def __init__(self):
        """Initialize the security middleware."""
        self.rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_window = 60.0  # 1 minute window
        self.rate_limit_max_requests = 100  # 100 requests per minute
        self.blocked_ips: Dict[str, float] = {}  # IP -> block expiry time
        self.block_duration = 300.0  # 5 minutes
        self.trusted_origins: List[str] = []
        self.trusted_ips: List[str] = ["127.0.0.1", "::1"]  # localhost
        self.trusted_networks: List[str] = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]  # private networks
        self.required_headers = []
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache"
        }
        self.content_security_policy = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' ws: wss:;"
        self.security_headers["Content-Security-Policy"] = self.content_security_policy
        
        # Load trusted origins from environment
        self._load_trusted_origins()
    
    def _load_trusted_origins(self) -> None:
        """Load trusted origins from environment variables."""
        trusted_origins_env = os.environ.get("WADE_TRUSTED_ORIGINS", "")
        if trusted_origins_env:
            self.trusted_origins = [origin.strip() for origin in trusted_origins_env.split(",")]
    
    def is_trusted_ip(self, ip: str) -> bool:
        """Check if an IP address is trusted."""
        if ip in self.trusted_ips:
            return True
        
        try:
            client_ip = ipaddress.ip_address(ip)
            for network in self.trusted_networks:
                if client_ip in ipaddress.ip_network(network):
                    return True
        except ValueError:
            # Invalid IP address
            return False
        
        return False
    
    def is_trusted_origin(self, origin: str) -> bool:
        """Check if an origin is trusted."""
        if not origin:
            return False
        
        if not self.trusted_origins:
            # If no trusted origins are configured, all origins are trusted
            return True
        
        try:
            parsed_origin = urlparse(origin)
            origin_host = parsed_origin.netloc
            
            for trusted_origin in self.trusted_origins:
                if trusted_origin == "*":
                    return True
                
                if trusted_origin == origin_host:
                    return True
                
                # Check for wildcard subdomains
                if trusted_origin.startswith("*."):
                    domain_suffix = trusted_origin[1:]  # Remove the *
                    if origin_host.endswith(domain_suffix):
                        return True
        except Exception:
            return False
        
        return False
    
    def check_rate_limit(self, ip: str) -> bool:
        """
        Check if an IP address has exceeded the rate limit.
        Returns True if the request is allowed, False if it should be blocked.
        """
        # Trusted IPs are not rate limited
        if self.is_trusted_ip(ip):
            return True
        
        # Check if IP is currently blocked
        if ip in self.blocked_ips:
            block_expiry = self.blocked_ips[ip]
            if time.time() < block_expiry:
                logger.warning(f"Blocked request from {ip} (IP is blocked)")
                return False
            else:
                # Block has expired
                del self.blocked_ips[ip]
        
        # Get request timestamps for this IP
        current_time = time.time()
        if ip not in self.rate_limits:
            self.rate_limits[ip] = []
        
        # Remove timestamps outside the window
        window_start = current_time - self.rate_limit_window
        self.rate_limits[ip] = [ts for ts in self.rate_limits[ip] if ts >= window_start]
        
        # Check if rate limit is exceeded
        if len(self.rate_limits[ip]) >= self.rate_limit_max_requests:
            # Block the IP
            self.blocked_ips[ip] = current_time + self.block_duration
            logger.warning(f"Blocked IP {ip} for {self.block_duration} seconds (rate limit exceeded)")
            return False
        
        # Add current timestamp
        self.rate_limits[ip].append(current_time)
        return True
    
    def validate_request_signature(self, request: Any) -> bool:
        """
        Validate the signature of a request.
        Returns True if the signature is valid, False otherwise.
        """
        # Get signature from headers
        signature = request.headers.get("X-Request-Signature")
        if not signature:
            # No signature provided
            return True  # Allow for now, but should be False in production
        
        # Get request body
        try:
            body = request.json()
        except Exception:
            # Invalid JSON or no body
            return False
        
        # Get timestamp from headers
        timestamp_str = request.headers.get("X-Request-Timestamp")
        if not timestamp_str:
            return False
        
        try:
            timestamp = float(timestamp_str)
        except ValueError:
            # Invalid timestamp
            return False
        
        # Check if timestamp is within acceptable range (5 minutes)
        current_time = time.time()
        if abs(current_time - timestamp) > 300:
            logger.warning(f"Request timestamp too old: {timestamp} (current: {current_time})")
            return False
        
        # Verify signature
        return request_signer.verify_signature(body, signature)
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate an authentication token.
        Returns the token payload if valid, None otherwise.
        """
        return token_manager.validate_token(token)
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers to add to responses."""
        return self.security_headers.copy()
    
    def set_content_security_policy(self, policy: str) -> None:
        """Set the Content-Security-Policy header."""
        self.content_security_policy = policy
        self.security_headers["Content-Security-Policy"] = policy
    
    def add_trusted_ip(self, ip: str) -> None:
        """Add an IP address to the trusted IPs list."""
        if ip not in self.trusted_ips:
            self.trusted_ips.append(ip)
    
    def add_trusted_network(self, network: str) -> None:
        """Add a network to the trusted networks list."""
        if network not in self.trusted_networks:
            try:
                # Validate network format
                ipaddress.ip_network(network)
                self.trusted_networks.append(network)
            except ValueError:
                logger.error(f"Invalid network format: {network}")
    
    def add_trusted_origin(self, origin: str) -> None:
        """Add an origin to the trusted origins list."""
        if origin not in self.trusted_origins:
            self.trusted_origins.append(origin)
    
    def clear_rate_limits(self) -> None:
        """Clear all rate limit data."""
        self.rate_limits.clear()
        self.blocked_ips.clear()

# Create singleton instance
security_middleware = SecurityMiddleware()