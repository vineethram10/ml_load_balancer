# models/dqn/train_agent_fast.py
import numpy as np
import pandas as pd
import pickle
import json
import os
import time
from collections import deque
import matplotlib.pyplot as plt

# Import the lightweight agent
try:
    from models.dqn.agent_lw import LightweightDQN
except ImportError:
    from agent_lw import LightweightDQN

class FastLoadBalancerEnvironment:
    """Simplified and faster environment for training"""
    def __init__(self, workload_data):
        self.workload_data = workload_data
        self.current_step = 0
        self.current_resources = 5
        self.min_resources = 2
        self.max_resources = 20
        
        # Cache frequently accessed data
        self.data_length = len(workload_data)
        self.cached_metrics = {}
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_resources = 5
        self.cached_metrics = {}
        return self._get_state()
    
    def _get_state(self):
        """Get current environment state - simplified version"""
        if self.current_step >= self.data_length - 1:
            self.current_step = 0
        
        # Current metrics
        current_load = self.workload_data.iloc[self.current_step]
        
        # Simplified state representation
        cpu_utilization = min(100, current_load['cpu_util_percent'] / self.current_resources * 5)
        memory_utilization = min(100, current_load['mem_util_percent'] / self.current_resources * 5)
        
        # Simple response time estimation
        response_time = 50 * (1 + (cpu_utilization / 100) ** 2)
        
        state = np.array([
            cpu_utilization / 100,
            memory_utilization / 100,
            current_load['net_out'] / 1000,
            response_time / 1000,
            self.current_resources / self.max_resources,
        ])
        
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Apply action
        if action == 1:  # Scale up
            self.current_resources = min(self.max_resources, self.current_resources + 1)
        elif action == 2:  # Scale down
            self.current_resources = max(self.min_resources, self.current_resources - 1)
        
        # Move to next time step
        self.current_step += 1
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate simplified metrics
        current_load = self.workload_data.iloc[self.current_step % self.data_length]
        cpu_utilization = min(100, current_load['cpu_util_percent'] / self.current_resources * 5)
        response_time = 50 * (1 + (cpu_utilization / 100) ** 2)
        
        # Simplified metrics
        metrics = {
            'response_time_p95': response_time,
            'resource_cost': self.current_resources * 0.1,
            'cpu_utilization': cpu_utilization,
        }
        
        # Check if episode is done
        done = self.current_step >= min(288, self.data_length - 1)  # Max 288 steps (24 hours)
        
        return next_state, metrics, done

def train_fast_dqn(episodes=50, save_path='models/dqn/weights/'):
    """Train the lightweight DQN agent quickly"""
    print("Loading training data...")
    
    # Load workload data
    try:
        workload_data_dir = os.path.join("..\..", "data/alibaba_subset/")
        print(workload_data_dir)
        workload_full_path = os.path.abspath(workload_data_dir)
        print(workload_full_path)
        file_path = os.path.join(workload_full_path, "processed_data.csv")
        print(file_path)
        workload_data = pd.read_csv(file_path)
        print("Loaded Alibaba dataset")
    except:
        # Generate minimal synthetic data for testing
        print("Generating minimal synthetic data...")
        hours = 24
        timestamps = pd.date_range('2024-01-01', periods=hours*12, freq='5min')
        workload_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_util_percent': np.random.uniform(20, 80, size=len(timestamps)),
            'mem_util_percent': np.random.uniform(30, 70, size=len(timestamps)),
            'net_out': np.random.uniform(100, 1000, size=len(timestamps))
        })
    
    # Initialize environment
    env = FastLoadBalancerEnvironment(workload_data)
    
    # Initialize lightweight agent
    state_size = 5  # Simplified state
    action_size = 3
    cooldown_config = {
        'scale_up_cooldown': 60,
        'scale_down_cooldown': 300,
    }
    
    # Add cooldown features to state size
    state_size_with_cooldown = state_size + 5
    
    agent = LightweightDQN(state_size_with_cooldown, action_size, cooldown_config)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_times = []
    
    print(f"\nStarting fast training for {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        episode_start = time.time()
        state = env.reset()
        action_history = []
        total_reward = 0
        steps = 0
        
        while True:
            # Get state with cooldown features
            cooldown_state = agent.get_state_with_cooldown(state, action_history)
            
            # Choose action
            action = agent.act(cooldown_state)
            
            # Take action
            next_state, metrics, done = env.step(action)
            
            # Calculate reward
            reward = agent.calculate_reward(state, action, next_state, metrics)
            
            # Get next state with cooldown
            next_cooldown_state = agent.get_state_with_cooldown(next_state, action_history)
            
            # Remember experience
            agent.remember(cooldown_state, action, reward, next_cooldown_state, done)
            
            # Update action history
            action_history.append({
                'action': ['no_action', 'scale_up', 'scale_down'][action],
                'timestamp': steps * 300
            })
            agent.record_action(action, steps * 300)
            
            # Train the model
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update target model periodically
        if episode % 5 == 0:
            agent.update_target_model()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track metrics
        episode_time = time.time() - episode_start
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        training_times.append(episode_time)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
            avg_time = np.mean(training_times[-10:]) if len(training_times) >= 10 else episode_time
            total_time = time.time() - start_time
            print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Episode Time: {avg_time:.2f}s, Total Time: {total_time:.1f}s, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f} seconds!")
    
    # Save the trained model
    print("Saving model...")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    agent.model.save_weights(os.path.join(save_path, 'dqn_model.h5'))
    
    # Save model config
    model_config = {
        'state_size': state_size_with_cooldown,
        'action_size': action_size,
        'cooldown_config': cooldown_config,
        'epsilon': agent.epsilon,
        'training_episodes': episodes,
        'total_training_time': total_training_time
    }
    
    with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    

    print(f"Model saved to {save_path}")
    
    # Plot training results
    plot_fast_training_results(episode_rewards, training_times)
    
    return agent

def plot_fast_training_results(rewards, times):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, color='blue')
    if len(rewards) >= 10:
        rolling_mean = pd.Series(rewards).rolling(window=10).mean()
        ax1.plot(rolling_mean, color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards over Episodes')
    ax1.legend(['Episode Reward', '10-Episode Moving Average'])
    ax1.grid(True, alpha=0.3)
    
    # Plot training times
    ax2.plot(times, alpha=0.6, color='green')
    ax2.axhline(y=np.mean(times), color='red', linestyle='--', label=f'Average: {np.mean(times):.2f}s')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Time (seconds)')
    ax2.set_title('Training Time per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/dqn/fast_training_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Train the lightweight agent
    agent = train_fast_dqn(episodes=50)  # Reduced episodes for faster training
    
    print("\nTraining complete! The lightweight model should train much faster.")