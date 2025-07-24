"""
Configuration management for Cricket-Insight Agent.

This module handles all configuration settings including environment variables,
API keys, and system parameters with proper validation and defaults.
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CricketConfig(BaseModel):
    """
    Configuration class for Cricket-Insight Agent.
    
    Handles all configuration settings with validation and environment variable loading.
    """
    
    # LLM Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server base URL")
    llm_model: str = Field(default="llama3.1", description="Ollama model to use")
    llm_temperature: float = Field(default=0.1, description="LLM temperature for consistency", ge=0.0, le=2.0)
    
    # MCP Server Configuration
    mcp_uri: Optional[str] = Field(default=None, description="MongoDB MCP server URI")
    mongodb_uri: Optional[str] = Field(default=None, description="MongoDB connection string")
    mongodb_database: str = Field(default="cricket_db", description="MongoDB database name")
    
    # Security Settings
    read_only: bool = Field(default=True, description="Enforce read-only database access")
    max_time_ms: int = Field(default=3000, description="Maximum query execution time", gt=0)
    allow_disk_use: bool = Field(default=False, description="Allow disk usage for aggregations")
    max_documents: int = Field(default=1000, description="Maximum documents per query", gt=0)
    
    # Application Settings
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Vector Store Configuration (Optional)
    enable_vector_store: bool = Field(default=False, description="Enable Chroma vector store")
    chroma_persist_dir: str = Field(default="./vector_store/data", description="Chroma persistence directory")
    embeddings_model: str = Field(default="nomic-embed-text", description="Ollama embeddings model")
    vector_store_k: int = Field(default=3, description="Number of similar tools to retrieve", gt=0)
    
    # Performance Settings
    mongodb_pool_size: int = Field(default=10, description="MongoDB connection pool size", gt=0)
    mongodb_timeout_ms: int = Field(default=5000, description="MongoDB connection timeout", gt=0)
    
    class Config:
        """Pydantic configuration."""
        env_prefix = ""  # No prefix for environment variables
        case_sensitive = False
        
    def __init__(self, **data):
        """
        Initialize configuration with environment variable loading.
        
        Args:
            **data: Override values for configuration
        """
        # Load from environment variables if not provided
        env_data = {}
        
        # Ollama Configuration
        if "ollama_base_url" not in data:
            env_data["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if "llm_model" not in data:
            env_data["llm_model"] = os.getenv("LLM_MODEL", "llama3.1")
        
        if "llm_temperature" not in data:
            env_data["llm_temperature"] = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        
        # MCP Configuration
        if "mcp_uri" not in data:
            env_data["mcp_uri"] = os.getenv("MCP_URI")
        
        if "mongodb_uri" not in data:
            env_data["mongodb_uri"] = os.getenv("MONGODB_URI")
        
        if "mongodb_database" not in data:
            env_data["mongodb_database"] = os.getenv("MONGODB_DATABASE", "cricket_db")
        
        # Security Settings
        if "read_only" not in data:
            env_data["read_only"] = os.getenv("READ_ONLY", "true").lower() == "true"
        
        if "max_time_ms" not in data:
            env_data["max_time_ms"] = int(os.getenv("MAX_TIME_MS", "3000"))
        
        if "allow_disk_use" not in data:
            env_data["allow_disk_use"] = os.getenv("ALLOW_DISK_USE", "false").lower() == "true"
        
        if "max_documents" not in data:
            env_data["max_documents"] = int(os.getenv("MAX_DOCUMENTS", "1000"))
        
        # Application Settings
        if "verbose_logging" not in data:
            env_data["verbose_logging"] = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
        
        if "log_level" not in data:
            env_data["log_level"] = os.getenv("LOG_LEVEL", "INFO")
        
        # Vector Store Settings
        if "enable_vector_store" not in data:
            env_data["enable_vector_store"] = os.getenv("ENABLE_VECTOR_STORE", "false").lower() == "true"
        
        if "chroma_persist_dir" not in data:
            env_data["chroma_persist_dir"] = os.getenv("CHROMA_PERSIST_DIR", "./vector_store/data")
        
        if "embeddings_model" not in data:
            env_data["embeddings_model"] = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
        
        if "vector_store_k" not in data:
            env_data["vector_store_k"] = int(os.getenv("VECTOR_STORE_K", "3"))
        
        # Performance Settings
        if "mongodb_pool_size" not in data:
            env_data["mongodb_pool_size"] = int(os.getenv("MONGODB_POOL_SIZE", "10"))
        
        if "mongodb_timeout_ms" not in data:
            env_data["mongodb_timeout_ms"] = int(os.getenv("MONGODB_TIMEOUT_MS", "5000"))
        
        # Merge environment data with provided data
        merged_data = {**env_data, **data}
        
        super().__init__(**merged_data)
        
        # Configure logging
        self._configure_logging()
    
    @validator("ollama_base_url")
    def validate_ollama_base_url(cls, v):
        """Validate Ollama base URL."""
        if not v:
            raise ValueError("Ollama base URL is required")
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("Ollama base URL must start with 'http://' or 'https://'")
        return v
    
    @validator("llm_model")
    def validate_llm_model(cls, v):
        """Validate LLM model name."""
        valid_models = [
            "llama3.1", "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
            "llama3", "llama3:8b", "llama3:70b",
            "llama2", "llama2:7b", "llama2:13b", "llama2:70b",
            "codellama", "codellama:7b", "codellama:13b", "codellama:34b"
        ]
        if v not in valid_models:
            logger.warning(f"Model {v} not in known valid models: {valid_models}")
        return v
    
    @validator("mongodb_uri")
    def validate_mongodb_uri(cls, v):
        """Validate MongoDB URI format."""
        if v and not v.startswith("mongodb://") and not v.startswith("mongodb+srv://"):
            raise ValueError("MongoDB URI must start with 'mongodb://' or 'mongodb+srv://'")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def _configure_logging(self):
        """Configure logging based on settings."""
        numeric_level = getattr(logging, self.log_level, logging.INFO)
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.verbose_logging:
            # Enable debug logging for specific modules
            logging.getLogger("cricket_agent").setLevel(logging.DEBUG)
            logging.getLogger("mcp_server").setLevel(logging.DEBUG)
            logging.getLogger("analytics").setLevel(logging.DEBUG)
    
    def get_mongodb_config(self) -> Dict[str, Any]:
        """
        Get MongoDB-specific configuration.
        
        Returns:
            Dictionary with MongoDB configuration
        """
        return {
            "uri": self.mongodb_uri,
            "database": self.mongodb_database,
            "pool_size": self.mongodb_pool_size,
            "timeout_ms": self.mongodb_timeout_ms,
            "read_only": self.read_only,
            "max_time_ms": self.max_time_ms,
            "allow_disk_use": self.allow_disk_use,
            "max_documents": self.max_documents
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM-specific configuration.
        
        Returns:
            Dictionary with LLM configuration
        """
        return {
            "base_url": self.ollama_base_url,
            "model": self.llm_model,
            "temperature": self.llm_temperature
        }
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """
        Get vector store configuration.
        
        Returns:
            Dictionary with vector store configuration
        """
        return {
            "enabled": self.enable_vector_store,
            "persist_dir": self.chroma_persist_dir,
            "embeddings_model": self.embeddings_model,
            "k": self.vector_store_k
        }
    
    def validate_required_settings(self) -> bool:
        """
        Validate that all required settings are present.
        
        Returns:
            True if all required settings are valid
            
        Raises:
            ValueError: If required settings are missing
        """
        if not self.ollama_base_url:
            raise ValueError("Ollama base URL is required")
        
        # MCP URI is optional but recommended
        if not self.mcp_uri:
            logger.warning("MCP URI not configured - database access will be unavailable")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = self.dict()
        
        # Mask sensitive information - for Ollama, the URL is generally not sensitive
        # but we can mask it if it contains authentication tokens
        
        if config_dict.get("mongodb_uri"):
            # Mask password in URI
            uri = config_dict["mongodb_uri"]
            if "@" in uri:
                parts = uri.split("@")
                user_pass = parts[0].split("://")[1]
                if ":" in user_pass:
                    user = user_pass.split(":")[0]
                    config_dict["mongodb_uri"] = uri.replace(user_pass, f"{user}:***")
        
        return config_dict
    
    @classmethod
    def from_env(cls) -> "CricketConfig":
        """
        Create configuration instance from environment variables only.
        
        Returns:
            CricketConfig instance loaded from environment
        """
        return cls()
    
    @classmethod
    def for_development(cls) -> "CricketConfig":
        """
        Create configuration optimized for development.
        
        Returns:
            CricketConfig instance with development settings
        """
        return cls(
            verbose_logging=True,
            log_level="DEBUG",
            llm_temperature=0.0,  # More deterministic for testing
            max_documents=100,    # Smaller limits for faster dev
            ollama_base_url="http://localhost:11434"  # Default local Ollama
        )
    
    @classmethod
    def for_production(cls) -> "CricketConfig":
        """
        Create configuration optimized for production.
        
        Returns:
            CricketConfig instance with production settings
        """
        return cls(
            verbose_logging=False,
            log_level="INFO",
            llm_temperature=0.1,
            max_documents=1000,
            mongodb_pool_size=20,  # Higher pool for production
            ollama_base_url="http://localhost:11434"  # Default local Ollama
        )