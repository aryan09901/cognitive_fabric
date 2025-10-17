from .base import BaseConfig

class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    
    # Application
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Blockchain
    BLOCKCHAIN_RPC_URL: str = "http://localhost:8545"
    CONTRACT_ADDRESS: str = "0x5FbDB2315678afecb367f032d93F642f64180aa3"  # Local Hardhat
    PRIVATE_KEY: str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Hardhat #0
    
    # AI Models
    LLM_MODEL: str = "microsoft/DialoGPT-medium"  # Smaller model for development
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Database
    VECTOR_DB_URL: str = "localhost:8000"
    
    # IPFS
    IPFS_URL: str = "localhost:5001"
    
    # Reinforcement Learning
    RL_BUFFER_SIZE: int = 1000  # Smaller for development
    RL_BATCH_SIZE: int = 32
    
    # Agent Settings
    MAX_AGENT_MEMORY: int = 1000
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Logging
    LOG_LEVEL: str = "DEBUG"
    
    class Config:
        env_file = ".env.development"