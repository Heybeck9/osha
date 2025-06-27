"""
Security module for WADE autonomous development environment.
Provides request signing, encryption, and secure credential storage.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Dict, Any, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import uuid

# Secure credential storage
class CredentialManager:
    """Securely store and retrieve credentials for various services."""
    
    def __init__(self, storage_path: str = None):
        """Initialize the credential manager with a storage path."""
        self.storage_path = storage_path or os.path.join(os.path.expanduser("~"), ".wade", "credentials")
        self._ensure_storage_dir()
        self._master_key = self._get_or_create_master_key()
        self._fernet = self._create_fernet()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key."""
        key_path = os.path.join(os.path.dirname(self.storage_path), ".master.key")
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            # Generate a new master key
            key = Fernet.generate_key()
            # Save it securely with restricted permissions
            with open(key_path, "wb") as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Only owner can read/write
            return key
    
    def _create_fernet(self) -> Fernet:
        """Create a Fernet encryption instance with the master key."""
        return Fernet(self._master_key)
    
    def store_credential(self, service: str, credential_data: Dict[str, Any]) -> None:
        """Store credentials for a service."""
        # Encrypt the credential data
        encrypted_data = self._fernet.encrypt(json.dumps(credential_data).encode())
        
        # Save to file
        service_file = os.path.join(self.storage_path, f"{service}.cred")
        with open(service_file, "wb") as f:
            f.write(encrypted_data)
        os.chmod(service_file, 0o600)  # Only owner can read/write
    
    def get_credential(self, service: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials for a service."""
        service_file = os.path.join(self.storage_path, f"{service}.cred")
        if not os.path.exists(service_file):
            return None
        
        with open(service_file, "rb") as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = self._fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception:
            # Handle decryption errors
            return None
    
    def delete_credential(self, service: str) -> bool:
        """Delete credentials for a service."""
        service_file = os.path.join(self.storage_path, f"{service}.cred")
        if os.path.exists(service_file):
            os.remove(service_file)
            return True
        return False


# Request signing for API security
class RequestSigner:
    """Sign and verify API requests to prevent tampering and replay attacks."""
    
    def __init__(self, secret_key: str = None):
        """Initialize with a secret key or generate one."""
        self.secret_key = secret_key or self._generate_secret_key()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key."""
        return secrets.token_hex(32)
    
    def sign_request(self, payload: Dict[str, Any], timestamp: int = None) -> Tuple[Dict[str, Any], str]:
        """Sign a request payload and return the signature."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Add timestamp to prevent replay attacks
        payload_with_timestamp = {**payload, "timestamp": timestamp}
        
        # Create a canonical string representation
        canonical = json.dumps(payload_with_timestamp, sort_keys=True)
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return payload_with_timestamp, signature
    
    def verify_signature(self, payload: Dict[str, Any], signature: str, max_age_seconds: int = 300) -> bool:
        """Verify a request signature and check for replay attacks."""
        # Check if timestamp exists
        if "timestamp" not in payload:
            return False
        
        # Check for replay attacks
        current_time = int(time.time())
        if current_time - payload["timestamp"] > max_age_seconds:
            return False
        
        # Create a canonical string representation
        canonical = json.dumps(payload, sort_keys=True)
        
        # Verify HMAC signature
        expected_signature = hmac.new(
            self.secret_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)


# TLS certificate management for secure WebSocket connections
class TLSManager:
    """Manage TLS certificates for secure WebSocket connections."""
    
    def __init__(self, cert_dir: str = None):
        """Initialize with a certificate directory."""
        self.cert_dir = cert_dir or os.path.join(os.path.expanduser("~"), ".wade", "certs")
        os.makedirs(self.cert_dir, exist_ok=True)
        self.cert_path = os.path.join(self.cert_dir, "wade.crt")
        self.key_path = os.path.join(self.cert_dir, "wade.key")
    
    def ensure_certificates(self) -> Tuple[str, str]:
        """Ensure TLS certificates exist, generating if needed."""
        if not os.path.exists(self.cert_path) or not os.path.exists(self.key_path):
            self._generate_self_signed_cert()
        
        return self.cert_path, self.key_path
    
    def _generate_self_signed_cert(self) -> None:
        """Generate a self-signed certificate for development use."""
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import datetime
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Generate self-signed certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "WADE Autonomous Development"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                # Valid for 1 year
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("127.0.0.1"),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Write private key to file
            with open(self.key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            os.chmod(self.key_path, 0o600)  # Only owner can read/write
            
            # Write certificate to file
            with open(self.cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
        except ImportError:
            # Fallback to using OpenSSL command line if cryptography package is not available
            import subprocess
            
            # Generate private key
            subprocess.run([
                "openssl", "genrsa", "-out", self.key_path, "2048"
            ], check=True)
            os.chmod(self.key_path, 0o600)  # Only owner can read/write
            
            # Generate CSR
            subprocess.run([
                "openssl", "req", "-new", "-key", self.key_path, 
                "-out", os.path.join(self.cert_dir, "wade.csr"),
                "-subj", "/C=US/ST=CA/L=San Francisco/O=WADE Autonomous Development/CN=localhost"
            ], check=True)
            
            # Generate self-signed certificate
            subprocess.run([
                "openssl", "x509", "-req", "-days", "365", 
                "-in", os.path.join(self.cert_dir, "wade.csr"),
                "-signkey", self.key_path, "-out", self.cert_path
            ], check=True)


# Secure token generation and validation
class TokenManager:
    """Generate and validate secure tokens for authentication."""
    
    def __init__(self, secret_key: str = None):
        """Initialize with a secret key or generate one."""
        self.secret_key = secret_key or secrets.token_hex(32)
    
    def generate_token(self, user_id: str, expiry_seconds: int = 86400) -> str:
        """Generate a secure token for a user with expiry."""
        # Create token payload
        payload = {
            "user_id": user_id,
            "exp": int(time.time()) + expiry_seconds,
            "jti": str(uuid.uuid4())
        }
        
        # Encode payload
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        
        # Create signature
        signature = hmac.new(
            self.secret_key.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine payload and signature
        return f"{payload_b64}.{signature}"
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return the payload if valid."""
        try:
            # Split token into payload and signature
            payload_b64, signature = token.split(".")
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload_b64.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(expected_signature, signature):
                return None
            
            # Decode payload
            payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
            
            # Check expiry
            if payload.get("exp", 0) < time.time():
                return None
            
            return payload
        except Exception:
            return None


# Initialize default instances
credential_manager = CredentialManager()
request_signer = RequestSigner()
tls_manager = TLSManager()
token_manager = TokenManager()