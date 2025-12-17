# alerts/rules.py

from typing import Dict, List, Callable, Any
from datetime import datetime
import pandas as pd


class AlertRule:
    """Represents a single alert rule"""
    
    def __init__(self, name: str, condition: str, threshold: float, metric: str, enabled: bool = True):
        """
        Args:
            name: Alert name/description
            condition: Comparison operator ('>', '<', '>=', '<=', '==')
            threshold: Threshold value
            metric: Metric to check ('zscore', 'spread', 'corr', 'price_btc', 'price_eth')
            enabled: Whether alert is active
        """
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.metric = metric
        self.enabled = enabled
        self.last_triggered = None
        self.trigger_count = 0
    
    def check(self, data: Dict[str, Any]) -> bool:
        """Check if alert condition is met"""
        if not self.enabled:
            return False
        
        if self.metric not in data:
            return False
        
        value = data[self.metric]
        if pd.isna(value):
            return False
        
        # Evaluate condition
        if self.condition == '>':
            triggered = value > self.threshold
        elif self.condition == '<':
            triggered = value < self.threshold
        elif self.condition == '>=':
            triggered = value >= self.threshold
        elif self.condition == '<=':
            triggered = value <= self.threshold
        elif self.condition == '==':
            triggered = abs(value - self.threshold) < 0.0001  # Float comparison
        else:
            return False
        
        if triggered:
            self.last_triggered = datetime.now()
            self.trigger_count += 1
        
        return triggered
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'condition': self.condition,
            'threshold': self.threshold,
            'metric': self.metric,
            'enabled': self.enabled,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'trigger_count': self.trigger_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        rule = cls(
            name=data['name'],
            condition=data['condition'],
            threshold=data['threshold'],
            metric=data['metric'],
            enabled=data.get('enabled', True)
        )
        if data.get('last_triggered'):
            rule.last_triggered = datetime.fromisoformat(data['last_triggered'])
        rule.trigger_count = data.get('trigger_count', 0)
        return rule


class AlertManager:
    """Manages alert rules and checking"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules.append(rule)
    
    def remove_rule(self, name: str):
        """Remove an alert rule by name"""
        self.rules = [r for r in self.rules if r.name != name]
    
    def check_all(self, data: Dict[str, Any]) -> List[AlertRule]:
        """Check all rules and return triggered ones"""
        triggered = []
        for rule in self.rules:
            if rule.check(data):
                triggered.append(rule)
        return triggered
    
    def get_rules(self) -> List[AlertRule]:
        """Get all rules"""
        return self.rules
    
    def get_enabled_rules(self) -> List[AlertRule]:
        """Get only enabled rules"""
        return [r for r in self.rules if r.enabled]
    
    def to_dict_list(self) -> List[Dict]:
        """Convert all rules to dictionary list"""
        return [rule.to_dict() for rule in self.rules]
    
    def load_from_dict_list(self, data: List[Dict]):
        """Load rules from dictionary list"""
        self.rules = [AlertRule.from_dict(d) for d in data]

