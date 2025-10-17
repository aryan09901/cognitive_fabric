import asyncio
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import threading
from flask import Flask, render_template, jsonify, Response
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import pandas as pd

from .metrics import metrics_collector

class RealTimeDashboard:
    """
    Real-time monitoring dashboard for Cognitive Fabric
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 3000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        self.dashboard_data = {
            'query_metrics': [],
            'system_health': [],
            'agent_performance': [],
            'knowledge_stats': []
        }
        self.update_thread = None
        self.running = False
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/metrics/health')
        def health_metrics():
            health = metrics_collector.get_health_metrics()
            return jsonify(health)
        
        @self.app.route('/api/metrics/queries')
        def query_metrics():
            stats = metrics_collector.get_query_stats(time_window=300)
            return jsonify(stats)
        
        @self.app.route('/api/metrics/system')
        def system_metrics():
            # Get various system metrics
            cpu_metrics = metrics_collector.get_system_metrics('cpu')
            memory_metrics = metrics_collector.get_system_metrics('memory')
            network_metrics = metrics_collector.get_system_metrics('network')
            
            return jsonify({
                'cpu': cpu_metrics,
                'memory': memory_metrics,
                'network': network_metrics
            })
        
        @self.app.route('/api/metrics/agents')
        def agent_metrics():
            # This would query agent performance data
            return jsonify({
                'active_agents': 5,  # Mock data
                'average_reputation': 85.6,
                'total_interactions': 1247
            })
        
        @self.app.route('/api/metrics/knowledge')
        def knowledge_metrics():
            # This would query knowledge base stats
            return jsonify({
                'total_items': 15432,
                'average_verification': 0.78,
                'verified_items': 12045,
                'sources_count': 156
            })
        
        @self.app.route('/api/charts/query_volume')
        def query_volume_chart():
            """Generate query volume chart"""
            # Mock data for demonstration
            hours = list(range(24))
            volumes = [max(0, 50 + i * 10 - abs(i-12)*8) for i in hours]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=volumes,
                mode='lines+markers',
                name='Query Volume',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.update_layout(
                title='Query Volume (Last 24 Hours)',
                xaxis_title='Hour',
                yaxis_title='Queries',
                template='plotly_white'
            )
            
            return jsonify(json.loads(fig.to_json()))
        
        @self.app.route('/api/charts/verification_trend')
        def verification_trend_chart():
            """Generate verification score trend chart"""
            # Mock data
            days = list(range(7))
            scores = [0.72, 0.75, 0.78, 0.81, 0.79, 0.82, 0.85]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days,
                y=scores,
                mode='lines+markers',
                name='Verification Score',
                line=dict(color='#2ca02c', width=3)
            ))
            
            fig.update_layout(
                title='Verification Score Trend (Last 7 Days)',
                xaxis_title='Day',
                yaxis_title='Score',
                template='plotly_white',
                yaxis=dict(range=[0.5, 1.0])
            )
            
            return jsonify(json.loads(fig.to_json()))
    
    def start(self):
        """Start the dashboard"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()
        
        # Start Flask app
        self.app.run(host=self.host, port=self.port, debug=False)
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
    
    def _update_worker(self):
        """Background worker to update dashboard data"""
        while self.running:
            try:
                # Update query metrics
                query_stats = metrics_collector.get_query_stats(time_window=300)
                self.dashboard_data['query_metrics'].append({
                    'timestamp': time.time(),
                    'data': query_stats
                })
                
                # Keep only last 100 data points
                for key in self.dashboard_data:
                    if len(self.dashboard_data[key]) > 100:
                        self.dashboard_data[key] = self.dashboard_data[key][-100:]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Dashboard update error: {e}")
                time.sleep(10)

# Create and start dashboard
dashboard = RealTimeDashboard()

if __name__ == "__main__":
    dashboard.start()