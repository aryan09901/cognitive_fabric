import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Alert:
    """Represents an alert condition"""
    
    def __init__(self, 
                 name: str,
                 condition: callable,
                 severity: str,  # 'low', 'medium', 'high', 'critical'
                 message: str,
                 cooldown: int = 300):  # seconds
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message = message
        self.cooldown = cooldown
        self.last_triggered = None
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if alert should trigger"""
        if self.last_triggered and (datetime.now() - self.last_triggered).seconds < self.cooldown:
            return False
        
        try:
            result = self.condition(data)
            if result:
                self.last_triggered = datetime.now()
            return result
        except Exception as e:
            logger.error(f"Error evaluating alert condition {self.name}: {e}")
            return False

class AlertManager:
    """
    Manages alert conditions and notifications
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.notification_channels = []
        self.alert_history = []
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default system alerts"""
        
        # Low verification score alert
        low_verification_alert = Alert(
            name="low_verification_score",
            condition=lambda data: data.get('verification_score', 1.0) < 0.3,
            severity="medium",
            message="Low verification score detected: {verification_score}",
            cooldown=600  # 10 minutes
        )
        self.alerts.append(low_verification_alert)
        
        # High processing time alert
        slow_processing_alert = Alert(
            name="high_processing_time",
            condition=lambda data: data.get('processing_time', 0) > 10.0,
            severity="low",
            message="High processing time: {processing_time}s",
            cooldown=300  # 5 minutes
        )
        self.alerts.append(slow_processing_alert)
        
        # Agent reputation drop alert
        reputation_drop_alert = Alert(
            name="agent_reputation_drop",
            condition=lambda data: data.get('reputation_change', 0) < -20,
            severity="high",
            message="Significant reputation drop for agent {agent_id}: {reputation_change}",
            cooldown=900  # 15 minutes
        )
        self.alerts.append(reputation_drop_alert)
        
        # Blockchain connection alert
        blockchain_down_alert = Alert(
            name="blockchain_connection_down",
            condition=lambda data: not data.get('blockchain_connected', True),
            severity="critical",
            message="Blockchain connection lost",
            cooldown=60  # 1 minute
        )
        self.alerts.append(blockchain_down_alert)
        
        # Knowledge base growth alert
        knowledge_growth_alert = Alert(
            name="knowledge_base_growth",
            condition=lambda data: data.get('knowledge_growth_rate', 0) > 1000,
            severity="low",
            message="Rapid knowledge base growth: {knowledge_growth_rate} items/hour",
            cooldown=3600  # 1 hour
        )
        self.alerts.append(knowledge_growth_alert)
    
    async def check_alerts(self, system_data: Dict[str, Any]):
        """Check all alert conditions"""
        triggered_alerts = []
        
        for alert in self.alerts:
            if alert.should_trigger(system_data):
                # Format message with actual data
                try:
                    formatted_message = alert.message.format(**system_data)
                except KeyError:
                    formatted_message = alert.message
                
                alert_info = {
                    'name': alert.name,
                    'severity': alert.severity,
                    'message': formatted_message,
                    'timestamp': datetime.now(),
                    'data': system_data
                }
                
                triggered_alerts.append(alert_info)
                self.alert_history.append(alert_info)
                
                # Send notifications
                await self._send_notifications(alert_info)
        
        return triggered_alerts
    
    async def _send_notifications(self, alert_info: Dict[str, Any]):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            try:
                if channel['type'] == 'email':
                    await self._send_email_alert(alert_info, channel['config'])
                elif channel['type'] == 'webhook':
                    await self._send_webhook_alert(alert_info, channel['config'])
                elif channel['type'] == 'log':
                    self._send_log_alert(alert_info)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel['type']}: {e}")
    
    async def _send_email_alert(self, alert_info: Dict[str, Any], config: Dict[str, str]):
        """Send email alert"""
        try:
            message = MIMEMultipart()
            message['From'] = config['from_email']
            message['To'] = config['to_email']
            message['Subject'] = f"[{alert_info['severity'].upper()}] {alert_info['name']}"
            
            body = f"""
            Cognitive Fabric Alert
            
            Severity: {alert_info['severity']}
            Alert: {alert_info['name']}
            Message: {alert_info['message']}
            Time: {alert_info['timestamp']}
            
            System Data: {alert_info['data']}
            """
            
            message.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(message)
            
            logger.info(f"Email alert sent: {alert_info['name']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert_info: Dict[str, Any], config: Dict[str, str]):
        """Send webhook alert"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=alert_info,
                    headers=config.get('headers', {})
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent: {alert_info['name']}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_log_alert(self, alert_info: Dict[str, Any]):
        """Log alert to file"""
        logger.warning(
            f"ALERT [{alert_info['severity']}] {alert_info['name']}: {alert_info['message']}"
        )
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add a notification channel"""
        self.notification_channels.append({
            'type': channel_type,
            'config': config
        })
    
    def add_custom_alert(self, alert: Alert):
        """Add a custom alert"""
        self.alerts.append(alert)
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]
    
    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics"""
        recent_alerts = self.get_alert_history(hours)
        
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'most_frequent_alert': max(
                set([alert['name'] for alert in recent_alerts]),
                key=[alert['name'] for alert in recent_alerts].count
            ) if recent_alerts else None,
            'time_period_hours': hours
        }

# Global alert manager instance
alert_manager = AlertManager()