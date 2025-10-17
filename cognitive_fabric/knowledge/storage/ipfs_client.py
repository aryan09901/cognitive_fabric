import ipfshttpclient
import asyncio
import json
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

from config.base import config

logger = logging.getLogger(__name__)

@dataclass
class IPFSFile:
    hash: str
    size: int
    name: str
    metadata: Dict[str, Any]

class IPFSClient:
    """
    IPFS client for decentralized file storage
    """
    
    def __init__(self, host: str = "localhost", port: int = 5001):
        self.client = None
        self.host = host
        self.port = port
        self._connect()
    
    def _connect(self):
        """Connect to IPFS daemon"""
        try:
            self.client = ipfshttpclient.connect(f"/ip4/{self.host}/tcp/{self.port}")
            logger.info(f"Connected to IPFS daemon at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to IPFS: {e}")
            self.client = None
    
    async def upload_file(self, file_path: str, pin: bool = True) -> Optional[str]:
        """Upload a file to IPFS"""
        if not self.client:
            logger.error("IPFS client not connected")
            return None
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.add, file_path, pin
            )
            logger.info(f"Uploaded file to IPFS: {result['Hash']}")
            return result['Hash']
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return None
    
    async def upload_json(self, data: Dict[str, Any], pin: bool = True) -> Optional[str]:
        """Upload JSON data to IPFS"""
        if not self.client:
            logger.error("IPFS client not connected")
            return None
        
        try:
            # Convert to JSON string
            json_str = json.dumps(data, ensure_ascii=False)
            
            # Add to IPFS
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.add_str, json_str
            )
            
            if pin:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client.pin.add, result
                )
            
            logger.info(f"Uploaded JSON to IPFS: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to upload JSON: {e}")
            return None
    
    async def download_file(self, ipfs_hash: str, output_path: str) -> bool:
        """Download a file from IPFS"""
        if not self.client:
            logger.error("IPFS client not connected")
            return False
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.get, ipfs_hash, output_path
            )
            logger.info(f"Downloaded file from IPFS: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    async def download_json(self, ipfs_hash: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from IPFS"""
        if not self.client:
            logger.error("IPFS client not connected")
            return None
        
        try:
            # Get file content
            content = await asyncio.get_event_loop().run_in_executor(
                None, self.client.cat, ipfs_hash
            )
            
            # Parse JSON
            data = json.loads(content.decode('utf-8'))
            logger.info(f"Downloaded JSON from IPFS: {ipfs_hash}")
            return data
        except Exception as e:
            logger.error(f"Failed to download JSON: {e}")
            return None
    
    async def pin_content(self, ipfs_hash: str) -> bool:
        """Pin content to prevent garbage collection"""
        if not self.client:
            logger.error("IPFS client not connected")
            return False
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.pin.add, ipfs_hash
            )
            logger.info(f"Pinned content: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to pin content: {e}")
            return False
    
    async def unpin_content(self, ipfs_hash: str) -> bool:
        """Unpin content"""
        if not self.client:
            logger.error("IPFS client not connected")
            return False
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.pin.rm, ipfs_hash
            )
            logger.info(f"Unpinned content: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to unpin content: {e}")
            return False
    
    async def get_file_info(self, ipfs_hash: str) -> Optional[IPFSFile]:
        """Get information about a file"""
        if not self.client:
            logger.error("IPFS client not connected")
            return None
        
        try:
            file_stats = await asyncio.get_event_loop().run_in_executor(
                None, self.client.files.stat, f"/ipfs/{ipfs_hash}"
            )
            
            return IPFSFile(
                hash=ipfs_hash,
                size=file_stats['Size'],
                name=ipfs_hash,
                metadata={'type': file_stats['Type']}
            )
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return None
    
    async def list_pinned_files(self) -> List[IPFSFile]:
        """List all pinned files"""
        if not self.client:
            logger.error("IPFS client not connected")
            return []
        
        try:
            pinned = await asyncio.get_event_loop().run_in_executor(
                None, self.client.pin.ls
            )
            
            pinned_files = []
            for hash_key, pin_info in pinned.items():
                pinned_files.append(IPFSFile(
                    hash=hash_key,
                    size=0,  # Would need additional call to get size
                    name=hash_key,
                    metadata={'type': pin_info['Type']}
                ))
            
            return pinned_files
        except Exception as e:
            logger.error(f"Failed to list pinned files: {e}")
            return []
    
    async def check_connectivity(self) -> bool:
        """Check if IPFS daemon is reachable"""
        if not self.client:
            return False
        
        try:
            # Simple version check to test connectivity
            version = await asyncio.get_event_loop().run_in_executor(
                None, self.client.version
            )
            return bool(version)
        except Exception as e:
            logger.error(f"IPFS connectivity check failed: {e}")
            return False

# Global IPFS client instance
ipfs_client = IPFSClient(
    host=config.IPFS_URL.split(":")[0] if "://" not in config.IPFS_URL else config.IPFS_URL.split("://")[1].split(":")[0],
    port=int(config.IPFS_URL.split(":")[-1]) if ":" in config.IPFS_URL else 5001
)