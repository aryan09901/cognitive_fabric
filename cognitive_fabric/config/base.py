import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class BaseConfig(BaseSettings):
    """Base configuration for Cognitive Fabric"""
    
    # Application
    APP_NAME: str = "Cognitive Fabric"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Blockchain
    BLOCKCHAIN_RPC_URL: str = Field(..., env="BLOCKCHAIN_RPC_URL")
    CONTRACT_ADDRESS: str = Field(..., env="CONTRACT_ADDRESS")
    PRIVATE_KEY: str = Field(..., env="PRIVATE_KEY")
    GAS_LIMIT: int = 500000
    NETWORK_ID: int = 1337
    
    # AI Models
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.1"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    RL_ALGORITHM: str = "PPO"
    
    # Vector Database
    VECTOR_DB_URL: str = Field(..., env="VECTOR_DB_URL")
    VECTOR_DB_COLLECTION: str = "cognitive_fabric"
    
    # IPFS
    IPFS_URL: str = Field(..., env="IPFS_URL")
    IPFS_TIMEOUT: int = 30
    
    # Reinforcement Learning
    RL_LEARNING_RATE: float = 0.0003
    RL_GAMMA: float = 0.99
    RL_BATCH_SIZE: int = 64
    RL_BUFFER_SIZE: int = 10000
    
    # Agent Settings
    MAX_AGENT_MEMORY: int = 10000
    INITIAL_REPUTATION: int = 100
    KNOWLEDGE_REWARD: int = 5
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    RL_BUFFER_SIZE: int = 1000  # Smaller for development

class ProductionConfig(BaseConfig):
    """Production configuration"""
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    API_WORKERS: int = 8

def get_config(environment: Optional[str] = None) -> BaseConfig:
    """Get configuration based on environment"""
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
    }
    
    return configs.get(env, DevelopmentConfig)()

# Global config instance
config = get_config()