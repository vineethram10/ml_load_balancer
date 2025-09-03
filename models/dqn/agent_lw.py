# models/dqn/agent_lightweight.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import time

# Import CooldownManager from the same directory
try:
    from .cooldown_manager import CooldownManager
except ImportError:
    # If running as a script, use direct import
    from cooldown_manager import CooldownManager

class LightweightDQN:
    def __init__(self, state_size, action_size, cooldown_config):
        self.state_size = state_size
        self.action_size = action_size
        self.cooldown_config = cooldown_config
        
        # Hyperparameters - optimized for faster training
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98  # Faster decay
        self.learning_rate = 0.001
        self.batch_size = 16  # Smaller batch size
        
        # Experience replay - smaller memory
        self.memory = deque(maxlen=500)  # Reduced from 2000
        
        # Neural network - much simpler architecture
        self.model = self._build_lightweight_model()
        self.target_model = self._build_lightweight_model()
        
        # Initialize cooldown manager
        self.cooldown_manager = CooldownManager(cooldown_config)
        
    def _build_lightweight_model(self):
        """Build very lightweight DQN model"""
        model = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def get_state_with_cooldown(self, base_state, action_history):
        """Enhance state with cooldown information"""
        cooldown_features = self.cooldown_manager.get_cooldown_features(action_history)
        return np.concatenate([base_state, cooldown_features])
    
    def calculate_reward(self, state, action, next_state, metrics):
        """Simplified reward calculation"""
        # Base performance reward
        performance_reward = -metrics['response_time_p95'] / 100  # Normalize
        
        # Cost penalty
        cost_penalty = metrics['resource_cost'] * 0.1
        
        # Simplified cooldown penalty
        if action != 0:  # If not "no_action"
            cooldown_penalty = 0.2
        else:
            cooldown_penalty = 0
        
        return performance_reward - cost_penalty - cooldown_penalty
    
    def act(self, state):
        """Choose action with epsilon-greedy policy and cooldown masking"""
        # Get valid actions considering cooldown
        valid_actions = self.cooldown_manager.get_valid_actions()
        
        if np.random.random() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Predict Q-values
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid actions
        masked_q_values = np.copy(q_values)
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[i] = -np.inf
        
        return np.argmax(masked_q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch training data
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        # Predict Q-values for starting states
        current_q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def record_action(self, action, timestamp):
        """Record action in cooldown manager"""
        action_name = ['no_action', 'scale_up', 'scale_down'][action]
        self.cooldown_manager.record_action(action_name, timestamp)