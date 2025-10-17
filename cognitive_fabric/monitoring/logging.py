import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any
import structlog

def setup_logging(level: str = "INFO", json_format: bool = False):
    """
    Setup structured logging for Cognitive Fabric
    """
    if json_format:
        # Structured logging with JSON format
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        # Human-readable format
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('cognitive_fabric.log')
            ]
        )

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    """
    return structlog.get_logger(name)

class CognitiveFabricLogger:
    """
    Custom logger for Cognitive Fabric with specialized methods
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.name = name
    
    def agent_activity(self, agent_id: str, action: str, details: Dict[str, Any]):
        """Log agent activity"""
        self.logger.info(
            "agent_activity",
            agent_id=agent_id,
            action=action,
            details=details,
            logger_name=self.name
        )
    
    def query_processed(self, query: str, agent_id: str, processing_time: float, verification_score: float):
        """Log query processing"""
        self.logger.info(
            "query_processed",
            query=query[:100] + "..." if len(query) > 100 else query,
            agent_id=agent_id,
            processing_time=processing_time,
            verification_score=verification_score,
            logger_name=self.name
        )
    
    def blockchain_interaction(self, transaction_type: str, tx_hash: str, status: str, details: Dict[str, Any]):
        """Log blockchain interactions"""
        self.logger.info(
            "blockchain_interaction",
            transaction_type=transaction_type,
            transaction_hash=tx_hash,
            status=status,
            details=details,
            logger_name=self.name
        )
    
    def knowledge_added(self, knowledge_id: str, source: str, size: int, verification_score: float):
        """Log knowledge addition"""
        self.logger.info(
            "knowledge_added",
            knowledge_id=knowledge_id,
            source=source,
            size=size,
            verification_score=verification_score,
            logger_name=self.name
        )
    
    def system_health(self, component: str, status: str, metrics: Dict[str, Any]):
        """Log system health status"""
        self.logger.info(
            "system_health",
            component=component,
            status=status,
            metrics=metrics,
            logger_name=self.name
        )
    
    def error_occurred(self, error_type: str, component: str, message: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        self.logger.error(
            "error_occurred",
            error_type=error_type,
            component=component,
            message=message,
            context=context or {},
            logger_name=self.name
        )
    
    def performance_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Log performance metrics"""
        self.logger.info(
            "performance_metric",
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            logger_name=self.name
        )

# Setup default logging
setup_logging(level="INFO", json_format=False)

# Create module-specific loggers
agent_logger = CognitiveFabricLogger("agents")
api_logger = CognitiveFabricLogger("api")
blockchain_logger = CognitiveFabricLogger("blockchain")
knowledge_logger = CognitiveFabricLogger("knowledge")
nlp_logger = CognitiveFabricLogger("nlp")