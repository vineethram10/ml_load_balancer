import requests
import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

class ImprovedCooldownAnalyzer:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.results = {
            'static': [],
            'dynamic': []
        }
        
    def generate_workload_pattern(self, pattern_type='spike'):
        """Generate different workload patterns for testing"""
        if pattern_type == 'spike':
            # Sudden spike pattern with longer stable periods
            pattern = []
            for i in range(150):  # Increased from 100 to 150 for better analysis
                if i < 30 or i > 120:
                    pattern.append(random.randint(20, 30))
                elif 50 <= i <= 100:
                    pattern.append(random.randint(75, 85))
                else:
                    pattern.append(random.randint(45, 55))  # Transition zones
            return pattern
            
        elif pattern_type == 'gradual':
            # Gradual increase and decrease
            return [
                min(90, max(10, int(50 + 40 * np.sin(i * np.pi / 75))))
                for i in range(150)
            ]
        elif pattern_type == 'oscillating':
            # Rapid oscillations with varying intensity
            pattern = []
            for i in range(150):
                if i % 15 < 7:  # Changed from 10/5 to 15/7 for more realistic oscillations
                    pattern.append(random.randint(25, 40))
                else:
                    pattern.append(random.randint(65, 80))
            return pattern
        else:
            return [random.randint(10, 90) for _ in range(150)]
    
    def test_static_cooldown(self, cooldown_config, pattern_type='spike', run_id=1):
        """Test with static cooldown periods"""
        print(f"\n=== Static Cooldown Test - Run {run_id} ({pattern_type} pattern) ===")
        print(f"Config: Scale-up={cooldown_config['scale_up']}s, Scale-down={cooldown_config['scale_down']}s")
        
        workload = self.generate_workload_pattern(pattern_type)
        results = []
        
        # Track scaling events
        last_scale_up = -float('inf')
        last_scale_down = -float('inf')
        blocked_actions = 0
        successful_actions = 0
        oscillations = 0
        last_action = None
        
        for i, cpu_util in enumerate(workload):
            current_time = i
            
            # Prepare request
            payload = {
                "cpu_utilization": cpu_util,
                "memory_utilization": random.randint(30, 70),
                "request_rate": random.randint(100, 400),
                "p95_response_time": random.randint(200, 800),
                "active_connections": random.randint(10, 40)
            }
            
            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=5)
                decision = response.json()
                
                # Extract action from response
                action = decision.get('dqn_action', 0)
                routing = decision.get('routing', 'no_action')
                
                # Check if action would be blocked by cooldown
                action_blocked = False
                if action == 1 and (current_time - last_scale_up) < cooldown_config['scale_up']:
                    action_blocked = True
                    routing = 'no_action'
                    blocked_actions += 1
                elif action == 2 and (current_time - last_scale_down) < cooldown_config['scale_down']:
                    action_blocked = True
                    routing = 'no_action'
                    blocked_actions += 1
                else:
                    successful_actions += 1
                    if action == 1:
                        last_scale_up = current_time
                    elif action == 2:
                        last_scale_down = current_time
                
                # Check for oscillation
                if last_action and last_action != action and action != 0:
                    if (last_action == 1 and action == 2) or (last_action == 2 and action == 1):
                        oscillations += 1
                
                last_action = action if not action_blocked else 0
                
                results.append({
                    'timestamp': current_time,
                    'cpu_utilization': cpu_util,
                    'action': action,
                    'routing': routing,
                    'blocked': action_blocked,
                    'predicted_load': decision.get('predicted_load', [0])[0] if isinstance(decision.get('predicted_load', [0]), list) else decision.get('predicted_load', 0)
                })
                
            except Exception as e:
                print(f"Error at sample {i}: {e}")
            
            time.sleep(0.05)  # Small delay between requests
        
        # Calculate metrics
        total_scaling_attempts = sum(1 for r in results if r['action'] != 0)
        
        metrics = {
            'run_id': run_id,
            'pattern': pattern_type,
            'cooldown_type': 'static',
            'cooldown_config': cooldown_config,
            'total_samples': len(results),
            'blocked_actions': blocked_actions,
            'successful_actions': successful_actions,
            'blocking_rate': (blocked_actions / total_scaling_attempts * 100) if total_scaling_attempts > 0 else 0,
            'oscillation_count': oscillations,
            'oscillation_rate': oscillations / len(results) * 100,
            'results': results
        }
        
        self.results['static'].append(metrics)
        print(f"Completed: Blocked={blocked_actions}/{total_scaling_attempts} attempts, Oscillations={oscillations}")
        return metrics
    
    def test_dynamic_cooldown(self, base_config, pattern_type='spike', run_id=1):
        """Test with IMPROVED dynamic cooldown that adapts based on system behavior"""
        print(f"\n=== Dynamic Cooldown Test - Run {run_id} ({pattern_type} pattern) ===")
        print(f"Base Config: Scale-up={base_config['scale_up']}s, Scale-down={base_config['scale_down']}s")
        
        workload = self.generate_workload_pattern(pattern_type)
        results = []
        
        # Dynamic cooldown parameters - IMPROVED
        current_cooldown = base_config.copy()
        cooldown_history = []
        adjustment_factor = 1.0
        
        # Track scaling events
        last_scale_up = -float('inf')
        last_scale_down = -float('inf')
        blocked_actions = 0
        successful_actions = 0
        oscillations = 0
        last_action = None
        recent_actions = []
        
        # Improved parameters
        WINDOW_SIZE = 15  # Increased from 10
        MIN_ADJUSTMENT = 0.2  # More aggressive minimum (was 0.4)
        MAX_ADJUSTMENT = 2.5  # Higher maximum (was 2.0)
        OSCILLATION_HIGH_THRESHOLD = 2  # Lower threshold (was 3)
        OSCILLATION_LOW_THRESHOLD = 1  # Same
        ADJUSTMENT_RATE = 0.15  # Faster adaptation (was 0.1)
        
        for i, cpu_util in enumerate(workload):
            current_time = i
            recent_oscillations = 0
            
            # Improved oscillation detection
            if len(recent_actions) >= WINDOW_SIZE:
                # Count direction changes in recent actions
                for j in range(1, len(recent_actions)):
                    if recent_actions[j] != recent_actions[j-1]:
                        if recent_actions[j] != 0 and recent_actions[j-1] != 0:
                            recent_oscillations += 1
                
                # More sophisticated adjustment logic
                oscillation_rate = recent_oscillations / (WINDOW_SIZE / 2)
                
                if oscillation_rate > 0.4:  # High oscillation rate
                    # Increase cooldown more aggressively
                    adjustment_factor = min(MAX_ADJUSTMENT, adjustment_factor * (1 + ADJUSTMENT_RATE * 2))
                elif oscillation_rate > 0.2:  # Moderate oscillation
                    # Slight increase
                    adjustment_factor = min(MAX_ADJUSTMENT, adjustment_factor * (1 + ADJUSTMENT_RATE))
                elif oscillation_rate < 0.1:  # Very stable
                    # Decrease cooldown more aggressively
                    adjustment_factor = max(MIN_ADJUSTMENT, adjustment_factor * (1 - ADJUSTMENT_RATE * 1.5))
                else:  # Stable
                    # Gradual decrease
                    adjustment_factor = max(MIN_ADJUSTMENT, adjustment_factor * (1 - ADJUSTMENT_RATE))
                
                # Apply dynamic adjustment with pattern-specific logic
                if pattern_type == 'oscillating' and oscillation_rate > 0.3:
                    # Be more conservative for oscillating patterns
                    current_cooldown['scale_up'] = int(base_config['scale_up'] * adjustment_factor * 1.2)
                    current_cooldown['scale_down'] = int(base_config['scale_down'] * adjustment_factor * 1.2)
                else:
                    current_cooldown['scale_up'] = int(base_config['scale_up'] * adjustment_factor)
                    current_cooldown['scale_down'] = int(base_config['scale_down'] * adjustment_factor)
                
                # Keep only last WINDOW_SIZE actions
                recent_actions = recent_actions[-WINDOW_SIZE:]
            
            # Prepare request
            payload = {
                "cpu_utilization": cpu_util,
                "memory_utilization": random.randint(30, 70),
                "request_rate": random.randint(100, 400),
                "p95_response_time": random.randint(200, 800),
                "active_connections": random.randint(10, 40)
            }
            
            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=5)
                decision = response.json()
                
                # Extract action from response
                action = decision.get('dqn_action', 0)
                routing = decision.get('routing', 'no_action')
                
                # Check if action would be blocked by DYNAMIC cooldown
                action_blocked = False
                if action == 1 and (current_time - last_scale_up) < current_cooldown['scale_up']:
                    action_blocked = True
                    routing = 'no_action'
                    blocked_actions += 1
                elif action == 2 and (current_time - last_scale_down) < current_cooldown['scale_down']:
                    action_blocked = True
                    routing = 'no_action'
                    blocked_actions += 1
                else:
                    successful_actions += 1
                    if action == 1:
                        last_scale_up = current_time
                    elif action == 2:
                        last_scale_down = current_time
                
                # Track action for oscillation detection
                recent_actions.append(action if not action_blocked else 0)
                
                # Check for oscillation
                if last_action and last_action != action and action != 0:
                    if (last_action == 1 and action == 2) or (last_action == 2 and action == 1):
                        oscillations += 1
                
                last_action = action if not action_blocked else 0
                
                results.append({
                    'timestamp': current_time,
                    'cpu_utilization': cpu_util,
                    'action': action,
                    'routing': routing,
                    'blocked': action_blocked,
                    'predicted_load': decision.get('predicted_load', [0])[0] if isinstance(decision.get('predicted_load', [0]), list) else decision.get('predicted_load', 0),
                    'dynamic_cooldown_up': current_cooldown['scale_up'],
                    'dynamic_cooldown_down': current_cooldown['scale_down'],
                    'adjustment_factor': adjustment_factor,
                    'oscillations_window': recent_oscillations
                })
                
                cooldown_history.append({
                    'timestamp': current_time,
                    'scale_up_cooldown': current_cooldown['scale_up'],
                    'scale_down_cooldown': current_cooldown['scale_down'],
                    'adjustment_factor': adjustment_factor,
                    'oscillations_window': recent_oscillations
                })
                
            except Exception as e:
                print(f"Error at sample {i}: {e}")
            
            time.sleep(0.05)
        
        # Calculate metrics
        total_scaling_attempts = sum(1 for r in results if r['action'] != 0)
        
        metrics = {
            'run_id': run_id,
            'pattern': pattern_type,
            'cooldown_type': 'dynamic',
            'base_config': base_config,
            'total_samples': len(results),
            'blocked_actions': blocked_actions,
            'successful_actions': successful_actions,
            'blocking_rate': (blocked_actions / total_scaling_attempts * 100) if total_scaling_attempts > 0 else 0,
            'oscillation_count': oscillations,
            'oscillation_rate': oscillations / len(results) * 100,
            'cooldown_history': cooldown_history,
            'results': results
        }
        
        self.results['dynamic'].append(metrics)
        print(f"Completed: Blocked={blocked_actions}/{total_scaling_attempts} attempts, Oscillations={oscillations}")
        print(f"Final adjustment factor: {adjustment_factor:.2f}")
        return metrics
    
    def plot_cooldown_profiles(self):
        """Generate cooldown profile diagrams"""
        if not self.results['static'] or not self.results['dynamic']:
            print("No results to plot")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Improved Cooldown Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot static cooldown results
        for i, static_result in enumerate(self.results['static'][:3]):
            if i >= 3:
                break
            ax = axes[0, i]
            
            if not static_result['results']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"Static - Run {i+1} (No data)")
                continue
                
            df = pd.DataFrame(static_result['results'])
            
            # Plot CPU utilization and actions
            ax2 = ax.twinx()
            ax.plot(df['timestamp'], df['cpu_utilization'], 'b-', alpha=0.5, label='CPU Util')
            
            # Mark scaling actions
            scale_ups = df[df['routing'] == 'scale_up']
            scale_downs = df[df['routing'] == 'scale_down']
            blocked = df[df['blocked'] == True]
            
            ax2.scatter(scale_ups['timestamp'], [1]*len(scale_ups), c='green', marker='^', s=50, label='Scale Up')
            ax2.scatter(scale_downs['timestamp'], [0]*len(scale_downs), c='red', marker='v', s=50, label='Scale Down')
            ax2.scatter(blocked['timestamp'], [0.5]*len(blocked), c='orange', marker='x', s=30, label='Blocked')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('CPU Utilization (%)', color='b')
            ax2.set_ylabel('Scaling Actions', color='g')
            ax.set_title(f"Static - Run {i+1} ({static_result['pattern']})\nBlocking: {static_result['blocking_rate']:.1f}%")
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Plot dynamic cooldown results
        for i, dynamic_result in enumerate(self.results['dynamic'][:3]):
            if i >= 3:
                break
            ax = axes[1, i]
            
            if not dynamic_result['results']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"Dynamic - Run {i+1} (No data)")
                continue
                
            df = pd.DataFrame(dynamic_result['results'])
            
            # Plot CPU utilization and dynamic cooldown
            ax2 = ax.twinx()
            ax.plot(df['timestamp'], df['cpu_utilization'], 'b-', alpha=0.5, label='CPU Util')
            
            if 'dynamic_cooldown_up' in df.columns:
                ax2.plot(df['timestamp'], df['dynamic_cooldown_up'], 'g--', alpha=0.7, label='Cooldown (up)')
                ax2.plot(df['timestamp'], df['adjustment_factor'] * 20, 'k:', alpha=0.5, label='Adj Factor x20')
            
            # Mark scaling actions
            scale_ups = df[df['routing'] == 'scale_up']
            scale_downs = df[df['routing'] == 'scale_down']
            blocked = df[df['blocked'] == True]
            
            ax.scatter(scale_ups['timestamp'], scale_ups['cpu_utilization'], c='green', marker='^', s=50)
            ax.scatter(scale_downs['timestamp'], scale_downs['cpu_utilization'], c='red', marker='v', s=50)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('CPU Utilization (%)', color='b')
            ax2.set_ylabel('Cooldown Period (s)', color='g')
            ax.set_title(f"Dynamic - Run {i+1} ({dynamic_result['pattern']})\nBlocking: {dynamic_result['blocking_rate']:.1f}%")
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_cooldown_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_metrics(self):
        """Generate comparison table between static and dynamic strategies"""
        if not self.results['static'] and not self.results['dynamic']:
            print("No results to compare")
            return None
            
        comparison = {}
        
        if self.results['static']:
            comparison['Static'] = {
                'Avg Blocking Rate (%)': np.mean([r['blocking_rate'] for r in self.results['static']]),
                'Avg Oscillation Rate (%)': np.mean([r['oscillation_rate'] for r in self.results['static']]),
                'Total Blocked': sum([r['blocked_actions'] for r in self.results['static']]),
                'Total Successful': sum([r['successful_actions'] for r in self.results['static']])
            }
        
        if self.results['dynamic']:
            comparison['Dynamic'] = {
                'Avg Blocking Rate (%)': np.mean([r['blocking_rate'] for r in self.results['dynamic']]),
                'Avg Oscillation Rate (%)': np.mean([r['oscillation_rate'] for r in self.results['dynamic']]),
                'Total Blocked': sum([r['blocked_actions'] for r in self.results['dynamic']]),
                'Total Successful': sum([r['successful_actions'] for r in self.results['dynamic']])
            }
        
        if comparison:
            df = pd.DataFrame(comparison).T
            print("\n=== Cooldown Strategy Comparison ===")
            print(df.to_string())
            
            # Calculate improvements
            if 'Static' in comparison and 'Dynamic' in comparison:
                print("\n=== Performance Improvements ===")
                blocking_improvement = (comparison['Static']['Avg Blocking Rate (%)'] - 
                                      comparison['Dynamic']['Avg Blocking Rate (%)'])
                print(f"Blocking Rate Reduction: {blocking_improvement:.1f}%")
                
                oscillation_improvement = ((comparison['Static']['Avg Oscillation Rate (%)'] - 
                                          comparison['Dynamic']['Avg Oscillation Rate (%)']) / 
                                         comparison['Static']['Avg Oscillation Rate (%)'] * 100)
                print(f"Oscillation Reduction: {oscillation_improvement:.1f}%")
                
                success_improvement = ((comparison['Dynamic']['Total Successful'] - 
                                      comparison['Static']['Total Successful']) / 
                                     comparison['Static']['Total Successful'] * 100)
                print(f"Success Rate Improvement: {success_improvement:.1f}%")
            
            return df
        return None

# Main execution
if __name__ == "__main__":
    # API endpoint
    api_endpoint = "https://i83k7lo5g3.execute-api.us-east-1.amazonaws.com/dev/balance"
    
    # Initialize analyzer
    analyzer = ImprovedCooldownAnalyzer(api_endpoint)
    
    # IMPROVED Static cooldown configurations
    static_configs = [
        {'scale_up': 30, 'scale_down': 120},   # Conservative (reduced from 60/300)
        {'scale_up': 15, 'scale_down': 60},    # Moderate (reduced from 30/150)
        {'scale_up': 8, 'scale_down': 30}      # Aggressive (reduced from 15/60)
    ]
    
    # Test patterns
    patterns = ['spike', 'gradual', 'oscillating']
    
    # Run static cooldown tests
    print("\n" + "="*60)
    print("IMPROVED STATIC COOLDOWN STRATEGY TESTING")
    print("="*60)
    for i, (config, pattern) in enumerate(zip(static_configs, patterns), 1):
        try:
            analyzer.test_static_cooldown(config, pattern, run_id=i)
        except Exception as e:
            print(f"Failed static test {i}: {e}")
    
    # Run dynamic cooldown tests with BETTER base config
    print("\n" + "="*60)
    print("IMPROVED DYNAMIC COOLDOWN STRATEGY TESTING")
    print("="*60)
    base_config = {'scale_up': 30, 'scale_down': 120}  # More reasonable base
    for i, pattern in enumerate(patterns, 1):
        try:
            analyzer.test_dynamic_cooldown(base_config, pattern, run_id=i)
        except Exception as e:
            print(f"Failed dynamic test {i}: {e}")
    
    # Generate analysis
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    try:
        # Plot cooldown profiles
        analyzer.plot_cooldown_profiles()
        
        # Generate comparison metrics
        comparison_df = analyzer.generate_comparison_metrics()
        
        # Save results to file
        with open('improved_cooldown_results.json', 'w') as f:
            json.dump(analyzer.results, f, indent=2, default=str)
        
        print("\nResults saved to improved_cooldown_results.json")
        print("Plots saved to improved_cooldown_profiles.png")
    except Exception as e:
        print(f"Error generating analysis: {e}")
        
    print("\nAnalysis complete!")