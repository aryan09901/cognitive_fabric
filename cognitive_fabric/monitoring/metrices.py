import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import threading
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    request_id: str
    query: str
    processing_time: float
    agent_count: int
    verification_score: float
    timestamp: float

class MetricsCollector:
    """
    Advanced metrics collector for monitoring system performance
    """
    
    def __init__(self, retention_period: int = 3600):  # 1 hour retention
        self.retention_period = retention_period
        self.query_metrics = deque(maxlen=10000)  # Keep recent queries
        self.system_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Prometheus metrics
        self.requests_total = Counter('cognitive_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('cognitive_request_duration_seconds', 'Request duration')
        self.query_duration = Histogram('cognitive_query_duration_seconds', 'Query processing duration')
        self.verification_score = Histogram('cognitive_verification_score', 'Verification scores')
        self.agent_count = Histogram('cognitive_agent_count', 'Agents per query')
        self.memory_usage = Gauge('cognitive_memory_usage_bytes', 'Memory usage')
        self.active_agents = Gauge('cognitive_active_agents', 'Active agents')
        
        # Background cleanup task
        self.cleanup_task = None
        self.running = False
    
    def start(self):
        """Start metrics collection and background tasks"""
        self.running = True
        self.cleanup_task = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_task.start()
        logger.info("Metrics collector started")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.join(timeout=5)
        logger.info("Metrics collector stopped")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.requests_total.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.request_duration.observe(duration)
    
    def record_query_metrics(
        self, 
        request_id: str, 
        query: str, 
        processing_time: float,
        agent_count: int,
        verification_score: float
    ):
        """Record query processing metrics"""
        metrics = QueryMetrics(
            request_id=request_id,
            query=query,
            processing_time=processing_time,
            agent_count=agent_count,
            verification_score=verification_score,
            timestamp=time.time()
        )
        
        self.query_metrics.append(metrics)
        
        # Update Prometheus metrics
        self.query_duration.observe(processing_time)
        self.verification_score.observe(verification_score)
        self.agent_count.observe(agent_count)
    
    def record_system_metric(self, metric_type: str, value: float, labels: Optional[Dict] = None):
        """Record system-level metrics"""
        key = metric_type
        if labels:
            key += ":" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        
        self.system_metrics[key].append((time.time(), value))
    
    def get_query_stats(self, time_window: int = 300) -> Dict[str, Any]:
        """Get query statistics for the given time window"""
        now = time.time()
        window_start = now - time_window
        
        recent_queries = [
            q for q in self.query_metrics 
            if q.timestamp >= window_start
        ]
        
        if not recent_queries:
            return {
                'total_queries': 0,
                'time_window': time_window
            }
        
        processing_times = [q.processing_time for q in recent_queries]
        verification_scores = [q.verification_score for q in recent_queries]
        agent_counts = [q.agent_count for q in recent_queries]
        
        return {
            'total_queries': len(recent_queries),
            'time_window': time_window,
            'processing_time': {
                'mean': statistics.mean(processing_times),
                'median': statistics.median(processing_times),
                'p95': np.percentile(processing_times, 95) if len(processing_times) > 1 else processing_times[0],
                'min': min(processing_times),
                'max': max(processing_times)
            },
            'verification_score': {
                'mean': statistics.mean(verification_scores),
                'median': statistics.median(verification_scores),
                'p95': np.percentile(verification_scores, 95) if len(verification_scores) > 1 else verification_scores[0],
                'min': min(verification_scores),
                'max': max(verification_scores)
            },
            'agent_utilization': {
                'mean': statistics.mean(agent_counts),
                'median': statistics.median(agent_counts),
                'max': max(agent_counts)
            }
        }
    
    def get_system_metrics(self, metric_type: str, time_window: int = 300) -> List[Dict[str, Any]]:
        """Get system metrics for the given type and time window"""
        now = time.time()
        window_start = now - time_window
        
        metrics = []
        for key in list(self.system_metrics.keys()):
            if key.startswith(metric_type):
                values = [
                    (ts, val) for ts, val in self.system_metrics[key]
                    if ts >= window_start
                ]
                if values:
                    metrics.append({
                        'key': key,
                        'values': values,
                        'latest': values[-1][1] if values else 0
                    })
        
        return metrics
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for system monitoring"""
        query_stats = self.get_query_stats(time_window=300)  # 5 minutes
        
        # Calculate system health score
        health_indicators = []
        
        # Query volume health
        if query_stats['total_queries'] > 0:
            avg_processing_time = query_stats['processing_time']['mean']
            if avg_processing_time < 2.0:
                health_indicators.append(1.0)
            elif avg_processing_time < 5.0:
                health_indicators.append(0.7)
            else:
                health_indicators.append(0.3)
        
        # Verification quality health
        if query_stats['total_queries'] > 0:
            avg_verification = query_stats['verification_score']['mean']
            if avg_verification > 0.8:
                health_indicators.append(1.0)
            elif avg_verification > 0.6:
                health_indicators.append(0.7)
            else:
                health_indicators.append(0.3)
        
        # Overall health score
        health_score = statistics.mean(health_indicators) if health_indicators else 1.0
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.4 else 'unhealthy',
            'query_metrics': query_stats,
            'timestamp': time.time()
        }
    
    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(REGISTRY)
    
    def _cleanup_worker(self):
        """Background worker to clean up old metrics"""
        while self.running:
            try:
                now = time.time()
                cutoff = now - self.retention_period
                
                # Clean query metrics
                while self.query_metrics and self.query_metrics[0].timestamp < cutoff:
                    self.query_metrics.popleft()
                
                # Clean system metrics
                for key in list(self.system_metrics.keys()):
                    while (self.system_metrics[key] and 
                           self.system_metrics[key][0][0] < cutoff):
                        self.system_metrics[key].popleft()
                
                time.sleep(60)  # Clean every minute
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                time.sleep(10)

# Global metrics instance
metrics_collector = MetricsCollector()