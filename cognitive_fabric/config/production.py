from .base import BaseConfig

class ProductionConfig(BaseConfig):
    """Production configuration"""
    
    # Application
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Blockchain
    BLOCKCHAIN_RPC_URL: str = "https://polygon-rpc.com"  # Polygon Mainnet
    CONTRACT_ADDRESS: str = ""  # Set actual deployed contract address
    PRIVATE_KEY: str = ""  # Set production private key
    
    # AI Models
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.1"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Vector Database
    VECTOR_DB_URL: str = "vector-db:8000"  # Docker service name
    
    # IPFS
    IPFS_URL: str = "ipfs-node:5001"  # Docker service name
    
    # Reinforcement Learning
    RL_BUFFER_SIZE: int = 10000
    RL_BATCH_SIZE: int = 64
    
    # Agent Settings
    MAX_AGENT_MEMORY: int = 10000
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Security
    CORS_ORIGINS: list = [
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ]
    
    # Monitoring
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    
    class Config:
        env_file = ".env.production"