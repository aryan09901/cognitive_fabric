#!/usr/bin/env python3
"""
Production deployment script for Cognitive Fabric
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manage deployment of Cognitive Fabric system"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.root_dir = Path(__file__).parent.parent
        self.config = self._load_deployment_config()
    
    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        config_path = self.root_dir / "config" / "deployment.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'docker_compose_file': 'docker-compose.prod.yml',
                'services': ['api', 'blockchain-node', 'vector-db', 'ipfs-node'],
                'health_check_timeout': 300,
                'backup_enabled': True
            }
    
    def setup_environment(self):
        """Setup deployment environment"""
        logger.info("Setting up deployment environment...")
        
        # Create necessary directories
        directories = [
            self.root_dir / "data",
            self.root_dir / "logs",
            self.root_dir / "backups",
            self.root_dir / "models"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Set environment variables
        env_vars = {
            'ENVIRONMENT': self.environment,
            'PYTHONPATH': str(self.root_dir),
            'MODELS_DIR': str(self.root_dir / "models")
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("Environment setup completed")
    
    def build_services(self):
        """Build Docker services"""
        logger.info("Building Docker services...")
        
        compose_file = self.config.get('docker_compose_file', 'docker-compose.yml')
        compose_path = self.root_dir / compose_file
        
        if not compose_path.exists():
            logger.error(f"Docker compose file not found: {compose_path}")
            return False
        
        try:
            # Build services
            subprocess.run([
                'docker-compose', '-f', str(compose_path), 'build',
                '--parallel', '--no-cache'
            ], check=True, cwd=self.root_dir)
            
            logger.info("Services built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build services: {e}")
            return False
    
    def deploy_services(self):
        """Deploy services to production"""
        logger.info("Deploying services...")
        
        compose_file = self.config.get('docker_compose_file', 'docker-compose.yml')
        compose_path = self.root_dir / compose_file
        
        try:
            # Deploy services
            subprocess.run([
                'docker-compose', '-f', str(compose_path), 'up', '-d',
                '--force-recreate', '--remove-orphans'
            ], check=True, cwd=self.root_dir)
            
            logger.info("Services deployed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy services: {e}")
            return False
    
    def run_migrations(self):
        """Run database and blockchain migrations"""
        logger.info("Running migrations...")
        
        try:
            # Run blockchain migrations
            subprocess.run([
                'docker-compose', 'exec', 'blockchain-node',
                'npx', 'hardhat', 'run', 'scripts/deploy.js', '--network', 'localhost'
            ], check=True, cwd=self.root_dir)
            
            # Run database migrations (if any)
            # subprocess.run([
            #     'docker-compose', 'exec', 'api',
            #     'python', 'scripts/migrate_database.py'
            # ], check=True, cwd=self.root_dir)
            
            logger.info("Migrations completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run migrations: {e}")
            return False
    
    def health_check(self):
        """Perform health check on deployed services"""
        logger.info("Performing health check...")
        
        import time
        import requests
        
        timeout = self.config.get('health_check_timeout', 300)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get('http://localhost:8000/health', timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        logger.info("All services are healthy")
                        return True
                
                logger.info("Services not fully healthy yet, waiting...")
                time.sleep(10)
                
            except requests.RequestException as e:
                logger.info(f"Health check failed: {e}, retrying...")
                time.sleep(10)
        
        logger.error("Health check timeout reached")
        return False
    
    def run_tests(self):
        """Run deployment tests"""
        logger.info("Running deployment tests...")
        
        try:
            # Run integration tests
            result = subprocess.run([
                'pytest', 'tests/test_integration.py', '-v',
                '--tb=short'
            ], cwd=self.root_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Deployment tests passed")
                return True
            else:
                logger.error(f"Deployment tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False
    
    def create_backup(self):
        """Create deployment backup"""
        if not self.config.get('backup_enabled', True):
            return True
        
        logger.info("Creating deployment backup...")
        
        backup_dir = self.root_dir / "backups" / f"deployment_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup important directories
            important_dirs = ['config', 'scripts', 'blockchain/contracts']
            
            for dir_name in important_dirs:
                src_dir = self.root_dir / dir_name
                if src_dir.exists():
                    subprocess.run([
                        'cp', '-r', str(src_dir), str(backup_dir / dir_name)
                    ], check=True)
            
            # Backup docker-compose file
            compose_file = self.config.get('docker_compose_file', 'docker-compose.yml')
            subprocess.run([
                'cp', str(self.root_dir / compose_file), str(backup_dir)
            ], check=True)
            
            logger.info(f"Backup created at: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def deploy(self):
        """Execute complete deployment pipeline"""
        logger.info(f"Starting deployment to {self.environment} environment")
        
        steps = [
            ("Environment setup", self.setup_environment),
            ("Backup creation", self.create_backup),
            ("Service building", self.build_services),
            ("Service deployment", self.deploy_services),
            ("Migrations", self.run_migrations),
            ("Health check", self.health_check),
            ("Deployment tests", self.run_tests)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"Executing: {step_name}")
            try:
                success = step_func()
                if not success:
                    failed_steps.append(step_name)
                    logger.error(f"Step failed: {step_name}")
                    break
                else:
                    logger.info(f"Completed: {step_name}")
            except Exception as e:
                failed_steps.append(step_name)
                logger.error(f"Step failed with exception: {step_name} - {e}")
                break
        
        if failed_steps:
            logger.error(f"Deployment failed at steps: {failed_steps}")
            return False
        else:
            logger.info("Deployment completed successfully!")
            return True

def main():
    parser = argparse.ArgumentParser(description="Deploy Cognitive Fabric")
    parser.add_argument(
        '--environment', 
        choices=['development', 'staging', 'production'],
        default='production',
        help='Deployment environment'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip deployment tests'
    )
    
    args = parser.parse_args()
    
    deployer = DeploymentManager(environment=args.environment)
    
    if args.skip_tests:
        deployer.config['skip_tests'] = True
    
    success = deployer.deploy()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()