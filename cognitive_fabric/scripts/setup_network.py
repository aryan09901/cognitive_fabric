#!/usr/bin/env python3
"""
Network setup script for Cognitive Fabric
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def setup_blockchain():
    """Setup local blockchain network"""
    print("🔗 Setting up local blockchain network...")
    
    blockchain_dir = Path(__file__).parent.parent / "blockchain"
    
    # Install dependencies
    print("Installing blockchain dependencies...")
    if not run_command("npm install", cwd=blockchain_dir):
        return False
    
    # Start local blockchain
    print("Starting local blockchain node...")
    if not run_command("npx hardhat node", cwd=blockchain_dir, background=True):
        return False
    
    # Wait for node to start
    time.sleep(5)
    
    # Deploy contracts
    print("Deploying smart contracts...")
    if not run_command("npx hardhat run scripts/deploy.js --network localhost", cwd=blockchain_dir):
        return False
    
    print("✅ Blockchain network setup completed")
    return True

def setup_vector_database():
    """Setup vector database"""
    print("📚 Setting up vector database...")
    
    # Start ChromaDB
    if not run_command("docker run -d -p 8000:8000 --name chromadb chromadb/chroma"):
        return False
    
    # Wait for startup
    time.sleep(3)
    
    print("✅ Vector database setup completed")
    return True

def setup_ipfs():
    """Setup IPFS node"""
    print("🌐 Setting up IPFS node...")
    
    # Start IPFS daemon
    if not run_command("docker run -d -p 5001:5001 -p 8080:8080 --name ipfs-node ipfs/kubo:latest"):
        return False
    
    # Wait for startup
    time.sleep(5)
    
    print("✅ IPFS node setup completed")
    return True

def setup_environment():
    """Setup environment variables"""
    print("⚙️ Setting up environment...")
    
    env_content = """
# Cognitive Fabric Development Environment
ENVIRONMENT=development

# Blockchain
BLOCKCHAIN_RPC_URL=http://localhost:8545
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80

# AI Models
LLM_MODEL=microsoft/DialoGPT-medium
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Database
VECTOR_DB_URL=localhost:8000

# IPFS
IPFS_URL=localhost:5001

# API
API_HOST=0.0.0.0
API_PORT=8000
"""
    
    env_path = Path(__file__).parent.parent / ".env"
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Environment setup completed")
    return True

def check_dependencies():
    """Check system dependencies"""
    print("🔍 Checking system dependencies...")
    
    dependencies = [
        ("python", "python --version"),
        ("node", "node --version"),
        ("npm", "npm --version"),
        ("docker", "docker --version"),
    ]
    
    for dep_name, cmd in dependencies:
        if run_command(cmd):
            print(f"✅ {dep_name} is installed")
        else:
            print(f"❌ {dep_name} is not installed")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Cognitive Fabric Network Setup")
    print("=" * 50)
    
    if not check_dependencies():
        print("❌ Please install missing dependencies")
        sys.exit(1)
    
    steps = [
        ("Environment Setup", setup_environment),
        ("Blockchain Network", setup_blockchain),
        ("Vector Database", setup_vector_database),
        ("IPFS Node", setup_ipfs),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📦 Step: {step_name}")
        if not step_func():
            print(f"❌ {step_name} failed")
            sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the API: python -m uvicorn api.main:app --reload")
    print("2. Run demo: python scripts/demo.py")
    print("3. Check health: curl http://localhost:8000/health")

if __name__ == "__main__":
    main()