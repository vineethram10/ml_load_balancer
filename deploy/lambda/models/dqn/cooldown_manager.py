# models/dqn/cooldown_manager.py
import time
import numpy as np
from collections import deque

class CooldownManager:
    def __init__(self, config):
        self.config = config
        self.action_history = deque(maxlen=100)
        self.last_action_times = {}
        
    def get_cooldown_features(self, action_history):
        """Extract cooldown-related features"""
        features = []
        
        # Time since last scale up/down
        features.append(self.time_since_action('scale_up'))
        features.append(self.time_since_action('scale_down'))
        
        # Cooldown progress (0-1)
        features.append(self.get_cooldown_progress('scale_up'))
        features.append(self.get_cooldown_progress('scale_down'))
        
        # Recent scaling frequency
        features.append(self.get_scaling_frequency(window=300))  # 5 min
        
        return np.array(features)
    
    def time_since_action(self, action_name):
        """Get time since last action of given type"""
        if action_name not in self.last_action_times:
            return float('inf')  # Never performed this action
        
        return time.time() - self.last_action_times[action_name]
    
    def time_since_last_action(self, action):
        """Get time since last action (for reward calculation)"""
        action_names = ['no_action', 'scale_up', 'scale_down']
        if 0 <= action < len(action_names):
            action_name = action_names[action]
            return self.time_since_action(action_name)
        return float('inf')
    
    def get_cooldown_progress(self, action_name):
        """Get cooldown progress (0-1, where 1 means cooldown complete)"""
        time_since = self.time_since_action(action_name)
        cooldown_period = self.config.get(f'{action_name}_cooldown', 60)
        
        if time_since == float('inf'):
            return 1.0  # No cooldown active
        
        progress = min(1.0, time_since / cooldown_period)
        return progress
    
    def get_scaling_frequency(self, window=300):
        """Calculate scaling frequency within time window"""
        current_time = time.time()
        recent_actions = [
            a for a in self.action_history 
            if current_time - a['timestamp'] < window and a['action'] != 'no_action'
        ]
        
        # Return actions per minute
        if window > 0:
            return len(recent_actions) / (window / 60.0)
        return 0.0
    
    def get_valid_actions(self):
        """Return list of valid actions considering cooldown"""
        valid_actions = []
        current_time = time.time()
        
        for action_id, action_name in enumerate(['no_action', 'scale_up', 'scale_down']):
            if action_name == 'no_action':
                valid_actions.append(action_id)
            else:
                last_time = self.last_action_times.get(action_name, 0)
                cooldown = self.config.get(f'{action_name}_cooldown', 60)
                
                if current_time - last_time >= cooldown:
                    valid_actions.append(action_id)
        
        return valid_actions
    
    def record_action(self, action_name, timestamp):
        """Record an action with its timestamp"""
        self.action_history.append({
            'action': action_name,
            'timestamp': timestamp
        })
        
        if action_name != 'no_action':
            self.last_action_times[action_name] = timestamp
    
    def calculate_oscillation_rate(self, window=300):
        """Calculate rate of opposing scaling actions"""
        current_time = time.time()
        recent_actions = [
            a for a in self.action_history 
            if current_time - a['timestamp'] < window
        ]
        
        if len(recent_actions) < 2:
            return 0.0
        
        oscillations = 0
        for i in range(1, len(recent_actions)):
            if (recent_actions[i]['action'] == 'scale_up' and 
                recent_actions[i-1]['action'] == 'scale_down') or \
               (recent_actions[i]['action'] == 'scale_down' and 
                recent_actions[i-1]['action'] == 'scale_up'):
                oscillations += 1
        
        return oscillations / (len(recent_actions) - 1)
    
    def get_status(self):
        """Get current cooldown status"""
        status = {
            'scale_up_cooldown_remaining': max(0, self.config.get('scale_up_cooldown', 60) - self.time_since_action('scale_up')),
            'scale_down_cooldown_remaining': max(0, self.config.get('scale_down_cooldown', 300) - self.time_since_action('scale_down')),
            'recent_oscillation_rate': self.calculate_oscillation_rate(),
            'scaling_frequency': self.get_scaling_frequency(),
            'valid_actions': self.get_valid_actions()
        }
        
        return status
    
    def can_scale(self, action_name):
        """Check if a specific scaling action is allowed"""
        if action_name == 'no_action':
            return True
        
        time_since = self.time_since_action(action_name)
        cooldown = self.config.get(f'{action_name}_cooldown', 60)
        
        return time_since >= cooldown
    
    def is_action_valid(self, action_id):
        """Check if an action ID is valid considering cooldowns"""
        return action_id in self.get_valid_actions()