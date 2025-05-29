"""
Enhanced Memory Manager with advanced security, content processing, and conflict resolution.
"""

import asyncio
import hashlib
import hmac
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from multimind.memory import (
    HybridMemory,
    VectorStoreMemory,
    FastWeightMemory
)

class SecurityLevel(Enum):
    BASIC = 1
    ENHANCED = 2
    HIGH = 3
    CRITICAL = 4

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    STRUCTURED = "structured"
    DOCUMENT = "document"
    DATASET = "dataset"
    MODEL = "model"
    CONFIG = "config"

@dataclass
class SecurityConfig:
    level: SecurityLevel
    encryption_key: Optional[bytes] = None
    require_authentication: bool = True
    require_authorization: bool = True
    audit_logging: bool = True
    rate_limiting: bool = True
    max_attempts: int = 3
    lockout_duration: timedelta = timedelta(minutes=15)

@dataclass
class ContentProcessor:
    name: str
    version: str
    supported_types: List[ContentType]
    processor: callable
    validator: callable
    conflict_resolver: callable

class EnhancedMemoryManager:
    def __init__(
        self,
        security_config: SecurityConfig,
        debug_mode: bool = False
    ):
        self.memory_system = HybridMemory(
            memories=[
                VectorStoreMemory(),
                FastWeightMemory(
                    input_size=768,
                    memory_size=1024
                )
            ]
        )
        self.security_config = security_config
        self.debug_mode = debug_mode
        self.content_processors: Dict[ContentType, ContentProcessor] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.lockouts: Dict[str, datetime] = {}
        
        # Initialize encryption if required
        if security_config.encryption_key:
            self.fernet = Fernet(self._derive_key(security_config.encryption_key))
        
        # Register default content processors
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Register default content processors."""
        self.register_processor(ContentProcessor(
            name="text_processor",
            version="1.0",
            supported_types=[ContentType.TEXT],
            processor=self._process_text,
            validator=self._validate_text,
            conflict_resolver=self._resolve_text_conflicts
        ))
        
        self.register_processor(ContentProcessor(
            name="code_processor",
            version="1.0",
            supported_types=[ContentType.CODE],
            processor=self._process_code,
            validator=self._validate_code,
            conflict_resolver=self._resolve_code_conflicts
        ))
        
        self.register_processor(ContentProcessor(
            name="document_processor",
            version="1.0",
            supported_types=[ContentType.DOCUMENT],
            processor=self._process_document,
            validator=self._validate_document,
            conflict_resolver=self._resolve_document_conflicts
        ))
        
        self.register_processor(ContentProcessor(
            name="dataset_processor",
            version="1.0",
            supported_types=[ContentType.DATASET],
            processor=self._process_dataset,
            validator=self._validate_dataset,
            conflict_resolver=self._resolve_dataset_conflicts
        ))
    
    async def add_content(
        self,
        content_id: str,
        content_type: ContentType,
        data: Any,
        metadata: Dict[str, Any],
        user_id: str
    ) -> None:
        """Add content with enhanced security and processing."""
        # Check rate limiting
        if not self._check_rate_limit(user_id):
            raise Exception("Rate limit exceeded")
        
        # Check lockout
        if self._is_locked_out(user_id):
            raise Exception("Account temporarily locked")
        
        # Get processor
        processor = self.content_processors.get(content_type)
        if not processor:
            raise ValueError(f"No processor available for content type: {content_type}")
        
        # Validate content
        if not processor.validator(data):
            raise ValueError("Content validation failed")
        
        # Process content
        processed_data = await processor.processor(data)
        
        # Encrypt if required
        if self.security_config.level.value >= SecurityLevel.ENHANCED.value:
            processed_data = self._encrypt_data(processed_data)
        
        # Add to memory system
        await self.memory_system.add_memory(
            memory_id=content_id,
            content=processed_data,
            metadata={
                "content_type": content_type.value,
                "user_id": user_id,
                "timestamp": datetime.now(),
                "security_level": self.security_config.level.value,
                **metadata
            }
        )
        
        # Log operation
        self._log_operation("add_content", user_id, content_id, metadata)
    
    async def update_content(
        self,
        content_id: str,
        updates: Dict[str, Any],
        user_id: str
    ) -> None:
        """Update content with conflict resolution."""
        # Get current content
        current = await self.memory_system.get_memory(content_id)
        
        # Get processor
        processor = self.content_processors.get(ContentType(current["metadata"]["content_type"]))
        if not processor:
            raise ValueError("No processor available for content type")
        
        # Resolve conflicts
        resolved_updates = await processor.conflict_resolver(current, updates)
        
        # Update content
        await self.memory_system.update_memory(
            memory_id=content_id,
            updates=resolved_updates
        )
        
        # Log operation
        self._log_operation("update_content", user_id, content_id, resolved_updates)
    
    def register_processor(self, processor: ContentProcessor) -> None:
        """Register a new content processor."""
        for content_type in processor.supported_types:
            self.content_processors[content_type] = processor
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        if not self.security_config.rate_limiting:
            return True
        
        now = datetime.now()
        user_attempts = self.rate_limits.get(user_id, [])
        
        # Remove old attempts
        user_attempts = [t for t in user_attempts if now - t < timedelta(minutes=1)]
        
        if len(user_attempts) >= 60:  # 60 attempts per minute
            return False
        
        user_attempts.append(now)
        self.rate_limits[user_id] = user_attempts
        return True
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        if user_id in self.lockouts:
            if datetime.now() < self.lockouts[user_id]:
                return True
            del self.lockouts[user_id]
        return False
    
    def _encrypt_data(self, data: Any) -> bytes:
        """Encrypt data using Fernet."""
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        elif isinstance(data, str):
            data = data.encode()
        return self.fernet.encrypt(data)
    
    def _decrypt_data(self, encrypted_data: bytes) -> Any:
        """Decrypt data using Fernet."""
        decrypted = self.fernet.decrypt(encrypted_data)
        try:
            return json.loads(decrypted)
        except json.JSONDecodeError:
            return decrypted.decode()
    
    def _derive_key(self, password: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'multimind_salt',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def _log_operation(self, operation: str, user_id: str, content_id: str, metadata: Dict) -> None:
        """Log operation with enhanced security."""
        if not self.security_config.audit_logging:
            return
        
        log_entry = {
            "timestamp": datetime.now(),
            "operation": operation,
            "user_id": user_id,
            "content_id": content_id,
            "metadata": metadata,
            "security_level": self.security_config.level.value
        }
        
        # Add hash for integrity
        log_entry["hash"] = self._generate_hash(log_entry)
        
        if self.debug_mode:
            print(f"DEBUG: {log_entry}")
    
    def _generate_hash(self, data: Dict) -> str:
        """Generate HMAC hash for data integrity."""
        message = json.dumps(data, sort_keys=True).encode()
        return hmac.new(
            self.security_config.encryption_key or b'default_key',
            message,
            hashlib.sha256
        ).hexdigest()
    
    # Content Processors
    async def _process_text(self, data: str) -> Dict:
        """Process text content."""
        return {
            "text": data,
            "embeddings": await self._get_text_embeddings(data),
            "metadata": {
                "length": len(data),
                "language": self._detect_language(data)
            }
        }
    
    async def _process_code(self, data: str) -> Dict:
        """Process code content."""
        return {
            "code": data,
            "ast": self._parse_ast(data),
            "embeddings": await self._get_code_embeddings(data),
            "metadata": {
                "language": self._detect_language(data),
                "complexity": self._calculate_complexity(data)
            }
        }
    
    async def _process_document(self, data: Dict) -> Dict:
        """Process document content."""
        return {
            "document": data,
            "embeddings": await self._get_document_embeddings(data),
            "metadata": {
                "type": data.get("type"),
                "version": data.get("version"),
                "sections": len(data.get("sections", []))
            }
        }
    
    async def _process_dataset(self, data: Dict) -> Dict:
        """Process dataset content."""
        return {
            "dataset": data,
            "embeddings": await self._get_dataset_embeddings(data),
            "metadata": {
                "rows": len(data.get("data", [])),
                "columns": len(data.get("columns", [])),
                "schema": data.get("schema")
            }
        }
    
    # Content Validators
    def _validate_text(self, data: str) -> bool:
        """Validate text content."""
        return isinstance(data, str) and len(data) > 0
    
    def _validate_code(self, data: str) -> bool:
        """Validate code content."""
        return isinstance(data, str) and self._is_valid_syntax(data)
    
    def _validate_document(self, data: Dict) -> bool:
        """Validate document content."""
        return (
            isinstance(data, dict) and
            "type" in data and
            "content" in data
        )
    
    def _validate_dataset(self, data: Dict) -> bool:
        """Validate dataset content."""
        return (
            isinstance(data, dict) and
            "data" in data and
            "columns" in data and
            "schema" in data
        )
    
    # Conflict Resolvers
    async def _resolve_text_conflicts(self, current: Dict, updates: Dict) -> Dict:
        """Resolve text content conflicts."""
        resolved = current.copy()
        
        if "text" in updates:
            # Implement diff-based merging
            resolved["text"] = self._merge_text(current["text"], updates["text"])
        
        if "metadata" in updates:
            resolved["metadata"] = {
                **current.get("metadata", {}),
                **updates["metadata"]
            }
        
        return resolved
    
    async def _resolve_code_conflicts(self, current: Dict, updates: Dict) -> Dict:
        """Resolve code content conflicts."""
        resolved = current.copy()
        
        if "code" in updates:
            # Implement AST-based merging
            resolved["code"] = self._merge_code(current["code"], updates["code"])
            resolved["ast"] = self._parse_ast(resolved["code"])
        
        if "metadata" in updates:
            resolved["metadata"] = {
                **current.get("metadata", {}),
                **updates["metadata"]
            }
        
        return resolved
    
    async def _resolve_document_conflicts(self, current: Dict, updates: Dict) -> Dict:
        """Resolve document content conflicts."""
        resolved = current.copy()
        
        if "document" in updates:
            # Implement section-based merging
            resolved["document"] = self._merge_document(
                current["document"],
                updates["document"]
            )
        
        if "metadata" in updates:
            resolved["metadata"] = {
                **current.get("metadata", {}),
                **updates["metadata"]
            }
        
        return resolved
    
    async def _resolve_dataset_conflicts(self, current: Dict, updates: Dict) -> Dict:
        """Resolve dataset content conflicts."""
        resolved = current.copy()
        
        if "dataset" in updates:
            # Implement schema-based merging
            resolved["dataset"] = self._merge_dataset(
                current["dataset"],
                updates["dataset"]
            )
        
        if "metadata" in updates:
            resolved["metadata"] = {
                **current.get("metadata", {}),
                **updates["metadata"]
            }
        
        return resolved
    
    # Helper Methods
    def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        # Implement language detection
        return "en"
    
    def _is_valid_syntax(self, code: str) -> bool:
        """Check if code has valid syntax."""
        # Implement syntax validation
        return True
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate code complexity."""
        # Implement complexity calculation
        return 1
    
    def _parse_ast(self, code: str) -> Dict:
        """Parse code into AST."""
        # Implement AST parsing
        return {}
    
    def _merge_text(self, current: str, update: str) -> str:
        """Merge text content."""
        # Implement text merging
        return f"{current}\n{update}"
    
    def _merge_code(self, current: str, update: str) -> str:
        """Merge code content."""
        # Implement code merging
        return f"{current}\n{update}"
    
    def _merge_document(self, current: Dict, update: Dict) -> Dict:
        """Merge document content."""
        # Implement document merging
        return {
            "type": current["type"],
            "content": f"{current['content']}\n{update['content']}"
        }
    
    def _merge_dataset(self, current: Dict, update: Dict) -> Dict:
        """Merge dataset content."""
        # Implement dataset merging
        return {
            "data": current["data"] + update["data"],
            "columns": current["columns"],
            "schema": current["schema"]
        }
    
    # Embedding Methods
    async def _get_text_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text."""
        # Implement text embedding
        return []
    
    async def _get_code_embeddings(self, code: str) -> List[float]:
        """Get embeddings for code."""
        # Implement code embedding
        return []
    
    async def _get_document_embeddings(self, document: Dict) -> List[float]:
        """Get embeddings for document."""
        # Implement document embedding
        return []
    
    async def _get_dataset_embeddings(self, dataset: Dict) -> List[float]:
        """Get embeddings for dataset."""
        # Implement dataset embedding
        return []

async def example_usage():
    """Demonstrate enhanced memory manager features."""
    # Create security config
    security_config = SecurityConfig(
        level=SecurityLevel.ENHANCED,
        encryption_key=b'your-secret-key',
        require_authentication=True,
        require_authorization=True,
        audit_logging=True,
        rate_limiting=True
    )
    
    # Create memory manager
    manager = EnhancedMemoryManager(
        security_config=security_config,
        debug_mode=True
    )
    
    # Add text content
    await manager.add_content(
        content_id="text_1",
        content_type=ContentType.TEXT,
        data="This is a sample text document.",
        metadata={"language": "en", "category": "documentation"},
        user_id="user1"
    )
    
    # Add code content
    await manager.add_content(
        content_id="code_1",
        content_type=ContentType.CODE,
        data="def hello_world():\n    print('Hello, World!')",
        metadata={"language": "python", "category": "example"},
        user_id="user1"
    )
    
    # Update content with conflict resolution
    await manager.update_content(
        content_id="text_1",
        updates={
            "text": "Updated text content",
            "metadata": {"version": "2.0"}
        },
        user_id="user1"
    )
    
    # Print debug information
    print("Memory manager initialized with enhanced security")
    print("Content processors registered:", len(manager.content_processors))
    print("Security level:", manager.security_config.level)

if __name__ == "__main__":
    asyncio.run(example_usage()) 