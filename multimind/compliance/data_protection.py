"""
Data protection implementation for compliance.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import hashlib
import hmac
import json
from cryptography.fernet import Fernet
from .governance import GovernanceConfig, DataCategory

class DataProtectionManager(BaseModel):
    """Data protection manager."""
    
    config: GovernanceConfig
    encryption_key: Optional[bytes] = None
    pseudonymization_salt: Optional[bytes] = None
    
    def __init__(self, **data):
        """Initialize data protection manager."""
        super().__init__(**data)
        if self.config.enable_encryption:
            self.encryption_key = Fernet.generate_key()
        if self.config.enable_pseudonymization:
            self.pseudonymization_salt = os.urandom(32)
    
    async def protect_data(
        self,
        data: Any,
        category: DataCategory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Protect data according to its category."""
        protected_data = {
            "original_data": data,
            "category": category,
            "metadata": metadata or {},
            "protection_applied": []
        }
        
        # Apply protection based on category
        if category in [DataCategory.PERSONAL, DataCategory.SENSITIVE]:
            if self.config.enable_encryption:
                protected_data["data"] = await self._encrypt_data(data)
                protected_data["protection_applied"].append("encryption")
            
            if self.config.enable_pseudonymization:
                protected_data["pseudonymized"] = await self._pseudonymize_data(data)
                protected_data["protection_applied"].append("pseudonymization")
        else:
            protected_data["data"] = data
        
        # Add integrity check
        protected_data["integrity_hash"] = self._generate_integrity_hash(protected_data)
        
        return protected_data
    
    async def unprotect_data(
        self,
        protected_data: Dict[str, Any]
    ) -> Any:
        """Unprotect data."""
        # Verify integrity
        if not self._verify_integrity(protected_data):
            raise ValueError("Data integrity check failed")
        
        # Decrypt if encrypted
        if "encryption" in protected_data["protection_applied"]:
            return await self._decrypt_data(protected_data["data"])
        
        return protected_data["data"]
    
    async def _encrypt_data(self, data: Any) -> str:
        """Encrypt data."""
        if not self.encryption_key:
            raise ValueError("Encryption not enabled")
        
        f = Fernet(self.encryption_key)
        data_bytes = json.dumps(data).encode()
        return f.encrypt(data_bytes).decode()
    
    async def _decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt data."""
        if not self.encryption_key:
            raise ValueError("Encryption not enabled")
        
        f = Fernet(self.encryption_key)
        decrypted_bytes = f.decrypt(encrypted_data.encode())
        return json.loads(decrypted_bytes.decode())
    
    async def _pseudonymize_data(self, data: Any) -> str:
        """Pseudonymize data."""
        if not self.pseudonymization_salt:
            raise ValueError("Pseudonymization not enabled")
        
        if isinstance(data, dict):
            # Pseudonymize dictionary values
            pseudonymized = {}
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    pseudonymized[key] = self._generate_pseudonym(value)
                else:
                    pseudonymized[key] = value
            return pseudonymized
        elif isinstance(data, (str, int, float)):
            return self._generate_pseudonym(data)
        else:
            return data
    
    def _generate_pseudonym(self, value: Union[str, int, float]) -> str:
        """Generate pseudonym for a value."""
        if not self.pseudonymization_salt:
            raise ValueError("Pseudonymization not enabled")
        
        value_str = str(value).encode()
        return hmac.new(
            self.pseudonymization_salt,
            value_str,
            hashlib.sha256
        ).hexdigest()
    
    def _generate_integrity_hash(self, data: Dict[str, Any]) -> str:
        """Generate integrity hash for data."""
        # Remove existing hash if present
        data_copy = data.copy()
        if "integrity_hash" in data_copy:
            del data_copy["integrity_hash"]
        
        # Generate hash
        data_str = json.dumps(data_copy, sort_keys=True).encode()
        return hashlib.sha256(data_str).hexdigest()
    
    def _verify_integrity(self, data: Dict[str, Any]) -> bool:
        """Verify data integrity."""
        if "integrity_hash" not in data:
            return False
        
        stored_hash = data["integrity_hash"]
        calculated_hash = self._generate_integrity_hash(data)
        
        return stored_hash == calculated_hash
    
    async def rotate_keys(self) -> None:
        """Rotate encryption and pseudonymization keys."""
        if self.config.enable_encryption:
            self.encryption_key = Fernet.generate_key()
        if self.config.enable_pseudonymization:
            self.pseudonymization_salt = os.urandom(32)
    
    async def get_protection_status(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get data protection status."""
        return {
            "is_encrypted": "encryption" in data.get("protection_applied", []),
            "is_pseudonymized": "pseudonymization" in data.get("protection_applied", []),
            "has_integrity_check": "integrity_hash" in data,
            "category": data.get("category"),
            "protection_applied": data.get("protection_applied", [])
        } 