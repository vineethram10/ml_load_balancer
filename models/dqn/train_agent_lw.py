# models/dqn/train_agent.py
import numpy as np
import pandas as pd
import pickle
import json
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Import the lightweight agent
try:
    from .agent_lw import LightweightDQN
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
        
        # Track environment statistics
        self.stats = {
            'resource_history': [],
            'cpu_utilization_history': [],
            'response_time_history': [],
            'action_history': [],
            'reward_history': []
        }
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_resources = 5
        self.cached_metrics = {}
        self.stats = {
            'resource_history': [],
            'cpu_utilization_history': [],
            'response_time_history': [],
            'action_history': [],
            'reward_history': []
        }
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
        
        # Track statistics
        self.stats['cpu_utilization_history'].append(cpu_utilization)
        self.stats['response_time_history'].append(response_time)
        self.stats['resource_history'].append(self.current_resources)
        
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
        # Track action
        self.stats['action_history'].append(action)
        
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

class MetricsTracker:
    """Track and visualize training metrics"""
    def __init__(self):
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'average_q_values': [],
            'epsilon_values': [],
            'loss_values': [],
            'action_distribution': [],
            'training_times': [],
            'cpu_utilization': [],
            'response_times': [],
            'resource_counts': [],
            'cooldown_violations': [],
            'oscillation_rates': []
        }
        
    def update(self, episode_data):
        """Update metrics with episode data"""
        for key, value in episode_data.items():
            if key in self.metrics:
                self.metrics[key].append(value)

def calculate_action_distribution(action_history):
    """Calculate action distribution"""
    if not action_history:
        return [0, 0, 0]
    
    action_counts = [0, 0, 0]  # no_action, scale_up, scale_down
    for action in action_history:
        if 0 <= action < 3:
            action_counts[action] += 1
    
    total = sum(action_counts)
    if total > 0:
        return [count/total * 100 for count in action_counts]
    return [0, 0, 0]

def calculate_oscillation_rate(action_history):
    """Calculate oscillation rate in actions"""
    if len(action_history) < 2:
        return 0
    
    oscillations = 0
    for i in range(1, len(action_history)):
        # Check for opposing actions (scale up followed by scale down or vice versa)
        if (action_history[i] == 1 and action_history[i-1] == 2) or \
           (action_history[i] == 2 and action_history[i-1] == 1):
            oscillations += 1
    
    return oscillations / (len(action_history) - 1)

def plot_comprehensive_training_results(metrics_tracker, save_path='models/dqn/'):
    """Create comprehensive training visualization"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create GridSpec for complex layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('DQN Agent Training Analysis', fontsize=20, fontweight='bold')
    
    # 1. Episode Rewards
    ax1 = fig.add_subplot(gs[0, :])
    rewards = metrics_tracker.metrics['episode_rewards']
    ax1.plot(rewards, alpha=0.4, color='blue', linewidth=1, label='Episode Reward')
    if len(rewards) >= 10:
        rolling_mean = pd.Series(rewards).rolling(window=10).mean()
        ax1.plot(rolling_mean, color='red', linewidth=2, label='10-Episode Moving Average')
    ax1.fill_between(range(len(rewards)), rewards, alpha=0.2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Training Rewards over Episodes', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(rewards) > 1:
        z = np.polyfit(range(len(rewards)), rewards, 1)
        p = np.poly1d(z)
        ax1.plot(range(len(rewards)), p(range(len(rewards))), "--", color='green', alpha=0.5, label='Trend')
    
    # 2. Epsilon Decay
    ax2 = fig.add_subplot(gs[1, 0])
    epsilon_values = metrics_tracker.metrics['epsilon_values']
    ax2.plot(epsilon_values, color='orange', linewidth=2)
    ax2.fill_between(range(len(epsilon_values)), epsilon_values, alpha=0.3, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon Value')
    ax2.set_title('Exploration Rate (ε) Decay')
    ax2.grid(True, alpha=0.3)
    
    # 3. Action Distribution Over Time
    ax3 = fig.add_subplot(gs[1, 1])
    action_dist = np.array(metrics_tracker.metrics['action_distribution'])
    if len(action_dist) > 0:
        ax3.stackplot(range(len(action_dist)), 
                     action_dist[:, 0], action_dist[:, 1], action_dist[:, 2],
                     labels=['No Action', 'Scale Up', 'Scale Down'],
                     colors=['#2E86AB', '#A23B72', '#F18F01'],
                     alpha=0.8)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Action Distribution (%)')
        ax3.set_title('Action Distribution Evolution')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    
    # 4. Training Time per Episode
    ax4 = fig.add_subplot(gs[1, 2])
    times = metrics_tracker.metrics['training_times']
    ax4.bar(range(len(times)), times, color='green', alpha=0.6)
    ax4.axhline(y=np.mean(times), color='red', linestyle='--', label=f'Avg: {np.mean(times):.2f}s')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Training Time per Episode')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Average CPU Utilization
    ax5 = fig.add_subplot(gs[2, 0])
    cpu_util = metrics_tracker.metrics['cpu_utilization']
    if cpu_util:
        ax5.plot(cpu_util, color='red', linewidth=2, alpha=0.7)
        ax5.fill_between(range(len(cpu_util)), cpu_util, alpha=0.3, color='red')
        ax5.axhline(y=70, color='green', linestyle='--', label='Target: 70%')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('CPU Utilization (%)')
        ax5.set_title('Average CPU Utilization')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Response Time Performance
    ax6 = fig.add_subplot(gs[2, 1])
    response_times = metrics_tracker.metrics['response_times']
    if response_times:
        ax6.plot(response_times, color='purple', linewidth=2)
        ax6.fill_between(range(len(response_times)), response_times, alpha=0.3, color='purple')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Avg Response Time (ms)')
        ax6.set_title('Response Time Evolution')
        ax6.grid(True, alpha=0.3)
    
    # 7. Resource Count Evolution
    ax7 = fig.add_subplot(gs[2, 2])
    resource_counts = metrics_tracker.metrics['resource_counts']
    if resource_counts:
        ax7.plot(resource_counts, color='brown', linewidth=2)
        ax7.fill_between(range(len(resource_counts)), resource_counts, alpha=0.3, color='brown')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Avg Resource Count')
        ax7.set_title('Resource Allocation Evolution')
        ax7.grid(True, alpha=0.3)
    
    # 8. Oscillation Rate
    ax8 = fig.add_subplot(gs[3, 0])
    oscillation_rates = metrics_tracker.metrics['oscillation_rates']
    if oscillation_rates:
        ax8.plot(oscillation_rates, color='red', linewidth=2)
        ax8.fill_between(range(len(oscillation_rates)), oscillation_rates, alpha=0.3, color='red')
        ax8.axhline(y=0.2, color='green', linestyle='--', label='Target: < 0.2')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Oscillation Rate')
        ax8.set_title('Action Oscillation Rate (Lower is Better)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. Performance Summary Statistics
    ax9 = fig.add_subplot(gs[3, 1:])
    ax9.axis('off')
    
    # Calculate statistics
    if len(rewards) > 0:
        stats_text = f"""
        TRAINING STATISTICS SUMMARY
        {'='*40}
        
        REWARDS:
        • Final Episode Reward: {rewards[-1]:.2f}
        • Best Episode Reward: {max(rewards):.2f}
        • Average Reward (last 10): {np.mean(rewards[-10:]):.2f}
        • Improvement: {((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100):.1f}%
        
        TRAINING:
        • Total Episodes: {len(rewards)}
        • Total Training Time: {sum(times):.1f}s
        • Average Episode Time: {np.mean(times):.2f}s
        • Final Epsilon: {epsilon_values[-1]:.3f}
        
        PERFORMANCE:
        • Final CPU Utilization: {cpu_util[-1]:.1f}% if cpu_util else 'N/A'
        • Final Response Time: {response_times[-1]:.1f}ms if response_times else 'N/A'
        • Final Oscillation Rate: {oscillation_rates[-1]:.3f} if oscillation_rates else 'N/A'
        
        ACTION DISTRIBUTION (Final):
        • No Action: {action_dist[-1][0]:.1f}% if len(action_dist) > 0 else 'N/A'
        • Scale Up: {action_dist[-1][1]:.1f}% if len(action_dist) > 0 else 'N/A'
        • Scale Down: {action_dist[-1][2]:.1f}% if len(action_dist) > 0 else 'N/A'
        """
        
        ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(os.path.join(save_path, 'comprehensive_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_learning_curves(metrics_tracker, save_path='models/dqn/'):
    """Plot detailed learning curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DQN Learning Curves', fontsize=16, fontweight='bold')
    
    # 1. Reward Learning Curve with Confidence Interval
    ax1 = axes[0, 0]
    rewards = metrics_tracker.metrics['episode_rewards']
    episodes = range(len(rewards))
    
    if len(rewards) >= 10:
        rolling_mean = pd.Series(rewards).rolling(window=10).mean()
        rolling_std = pd.Series(rewards).rolling(window=10).std()
        
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
        ax1.plot(episodes, rolling_mean, color='red', linewidth=2, label='Rolling Mean')
        ax1.fill_between(episodes, 
                         rolling_mean - rolling_std, 
                         rolling_mean + rolling_std,
                         alpha=0.2, color='red', label='±1 std')
    else:
        ax1.plot(episodes, rewards, color='blue', linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Reward Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Value Evolution
    ax2 = axes[0, 1]
    q_values = metrics_tracker.metrics.get('average_q_values', [])
    if q_values:
        ax2.plot(q_values, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Q-Value')
        ax2.set_title('Q-Value Evolution')
        ax2.grid(True, alpha=0.3)
    
    # 3. Loss Evolution
    ax3 = axes[1, 0]
    loss_values = metrics_tracker.metrics.get('loss_values', [])
    if loss_values:
        ax3.plot(loss_values, color='orange', linewidth=1, alpha=0.7)
        if len(loss_values) >= 10:
            rolling_loss = pd.Series(loss_values).rolling(window=10).mean()
            ax3.plot(range(len(rolling_loss)), rolling_loss, color='red', linewidth=2, label='Rolling Mean')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss Evolution')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Episode Length Evolution
    ax4 = axes[1, 1]
    lengths = metrics_tracker.metrics['episode_lengths']
    ax4.plot(lengths, color='purple', linewidth=2)
    ax4.fill_between(range(len(lengths)), lengths, alpha=0.3, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Episode Length (steps)')
    ax4.set_title('Episode Length Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_training_metrics(metrics_tracker, agent, save_path='models/dqn/'):
    """Save all training metrics to JSON"""
    metrics_data = {
        'training_summary': {
            'total_episodes': len(metrics_tracker.metrics['episode_rewards']),
            'total_training_time': sum(metrics_tracker.metrics['training_times']),
            'final_epsilon': agent.epsilon,
            'final_reward': metrics_tracker.metrics['episode_rewards'][-1] if metrics_tracker.metrics['episode_rewards'] else 0,
            'best_reward': max(metrics_tracker.metrics['episode_rewards']) if metrics_tracker.metrics['episode_rewards'] else 0,
            'average_reward_last_10': np.mean(metrics_tracker.metrics['episode_rewards'][-10:]) if len(metrics_tracker.metrics['episode_rewards']) >= 10 else 0
        },
        'performance_metrics': {
            'final_cpu_utilization': metrics_tracker.metrics['cpu_utilization'][-1] if metrics_tracker.metrics['cpu_utilization'] else 0,
            'final_response_time': metrics_tracker.metrics['response_times'][-1] if metrics_tracker.metrics['response_times'] else 0,
            'final_oscillation_rate': metrics_tracker.metrics['oscillation_rates'][-1] if metrics_tracker.metrics['oscillation_rates'] else 0,
            'average_resources': np.mean(metrics_tracker.metrics['resource_counts']) if metrics_tracker.metrics['resource_counts'] else 0
        },
        'full_metrics': {
            'episode_rewards': metrics_tracker.metrics['episode_rewards'],
            'episode_lengths': metrics_tracker.metrics['episode_lengths'],
            'epsilon_values': metrics_tracker.metrics['epsilon_values'],
            'training_times': metrics_tracker.metrics['training_times']
        }
    }
    
    os.makedirs(os.path.join(save_path, 'metrics'), exist_ok=True)
    with open(os.path.join(save_path, 'metrics', 'training_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=2, default=float)
    
    print(f"\n✅ Metrics saved to {save_path}metrics/training_metrics.json")

def train_fast_dqn(episodes=50, save_path='models/dqn/weights/'):
    """Train the lightweight DQN agent quickly with comprehensive metrics"""
    print("\n" + "="*60)
    print("STARTING DQN AGENT TRAINING")
    print("="*60)
    
    print("\n📊 Loading training data...")
    
    # Load workload data
    try:
        workload_data_dir = os.path.join("", "data/alibaba_subset/")
        print(workload_data_dir)
        workload_full_path = os.path.abspath(workload_data_dir)
        print(workload_full_path)
        file_path = os.path.join(workload_full_path, "processed_data.csv")
        print(file_path)
        workload_data = pd.read_csv(file_path)
        print(f"✅ Loaded Alibaba dataset: {workload_data.shape}")
        print(f"Data columns: {workload_data.columns.tolist()}")
    except:
        # Generate minimal synthetic data for testing
        print("⚠️ Generating minimal synthetic data...")
        hours = 24
        timestamps = pd.date_range('2024-01-01', periods=hours*12, freq='5min')
        workload_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_util_percent': np.random.uniform(20, 80, size=len(timestamps)),
            'mem_util_percent': np.random.uniform(30, 70, size=len(timestamps)),
            'net_out': np.random.uniform(100, 1000, size=len(timestamps))
        })
    
    # Initialize environment
    print("\n🌍 Initializing environment...")
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
    
    print(f"\n🤖 Initializing DQN agent...")
    print(f"   State size: {state_size_with_cooldown}")
    print(f"   Action size: {action_size}")
    print(f"   Cooldown config: {cooldown_config}")
    
    agent = LightweightDQN(state_size_with_cooldown, action_size, cooldown_config)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    print(f"\n🚀 Starting training for {episodes} episodes...")
    print("-" * 60)
    
    start_time = time.time()
    
    for episode in range(episodes):
        episode_start = time.time()
        state = env.reset()
        action_history = []
        total_reward = 0
        steps = 0
        episode_losses = []
        
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
        
        # Calculate episode metrics
        episode_metrics = {
            'episode_rewards': total_reward,
            'episode_lengths': steps,
            'epsilon_values': agent.epsilon,
            'training_times': episode_time,
            'action_distribution': calculate_action_distribution(env.stats['action_history']),
            'cpu_utilization': np.mean(env.stats['cpu_utilization_history']) if env.stats['cpu_utilization_history'] else 0,
            'response_times': np.mean(env.stats['response_time_history']) if env.stats['response_time_history'] else 0,
            'resource_counts': np.mean(env.stats['resource_history']) if env.stats['resource_history'] else 0,
            'oscillation_rates': calculate_oscillation_rate(env.stats['action_history'])
        }
        
        metrics_tracker.update(episode_metrics)
        
        # Print progress
        if episode % 5 == 0:
            avg_reward = np.mean(metrics_tracker.metrics['episode_rewards'][-10:]) if len(metrics_tracker.metrics['episode_rewards']) >= 10 else total_reward
            avg_time = np.mean(metrics_tracker.metrics['training_times'][-10:]) if len(metrics_tracker.metrics['training_times']) >= 10 else episode_time
            total_time = time.time() - start_time
            
            print(f"Episode {episode}/{episodes}")
            print(f"  Reward: {total_reward:.2f} (Avg last 10: {avg_reward:.2f})")
            print(f"  Steps: {steps}, Time: {episode_time:.2f}s")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  CPU Util: {episode_metrics['cpu_utilization']:.1f}%")
            print(f"  Oscillation Rate: {episode_metrics['oscillation_rates']:.3f}")
            print(f"  Total Training Time: {total_time:.1f}s")
            print("-" * 60)
    
    total_training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ TRAINING COMPLETED IN {total_training_time:.1f} SECONDS")
    print("="*60)
    
    # Save the trained model
    print("\n💾 Saving model and configuration...")
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
    
    print(f"✅ Model saved to {save_path}")
    
    # Generate comprehensive visualizations
    print("\n📊 Generating comprehensive visualizations...")
    os.makedirs('models/dqn/plots', exist_ok=True)
    
    # Plot comprehensive training results
    plot_comprehensive_training_results(metrics_tracker, save_path='models/dqn/')
    
    # Plot learning curves
    plot_learning_curves(metrics_tracker, save_path='models/dqn/')
    
    # Save metrics
    save_training_metrics(metrics_tracker, agent, save_path='models/dqn/')
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Total Episodes: {episodes}")
    print(f"✅ Training Duration: {total_training_time:.1f} seconds")
    print(f"✅ Final Reward: {metrics_tracker.metrics['episode_rewards'][-1]:.2f}")
    print(f"✅ Best Reward: {max(metrics_tracker.metrics['episode_rewards']):.2f}")
    print(f"✅ Average Reward (last 10): {np.mean(metrics_tracker.metrics['episode_rewards'][-10:]):.2f}")
    print(f"✅ Final Epsilon: {agent.epsilon:.3f}")
    print(f"✅ Final CPU Utilization: {metrics_tracker.metrics['cpu_utilization'][-1]:.1f}%")
    print(f"✅ Final Oscillation Rate: {metrics_tracker.metrics['oscillation_rates'][-1]:.3f}")
    print(f"✅ Plots saved to: models/dqn/plots/")
    print(f"✅ Metrics saved to: models/dqn/metrics/")
    print("="*60)
    
    return agent

if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    import tensorflow as tf
    
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    # Train the lightweight agent with comprehensive metrics
    agent = train_fast_dqn(episodes=50)  # Reduced episodes for faster training
    
    print("\n🎉 Training complete! Check the plots and metrics folders for detailed analysis.")