import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

class SyntheticWorkloadGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_workload(self, duration_hours=24, interval_minutes=5):
        """Generate synthetic workload with various patterns"""
        timestamps = []
        current_time = datetime.now()
        
        for i in range(int(duration_hours * 60 / interval_minutes)):
            timestamps.append(current_time + timedelta(minutes=i * interval_minutes))
        
        # Base load pattern (diurnal)
        time_of_day = np.array([t.hour + t.minute/60 for t in timestamps])
        base_load = 30 + 20 * np.sin((time_of_day - 6) * np.pi / 12)
        
        # Add weekly pattern
        day_of_week = np.array([t.weekday() for t in timestamps])
        weekly_factor = np.where(day_of_week < 5, 1.2, 0.8)  # Higher on weekdays
        
        # Add random spikes
        spikes = np.random.choice([0, 1], size=len(timestamps), p=[0.95, 0.05])
        spike_magnitude = np.random.uniform(20, 50, size=len(timestamps))
        
        # Combine patterns
        cpu_util = base_load * weekly_factor + spikes * spike_magnitude
        cpu_util = np.clip(cpu_util, 0, 100)
        
        # Generate correlated memory usage
        memory_util = cpu_util * 0.8 + np.random.normal(0, 5, size=len(timestamps))
        memory_util = np.clip(memory_util, 0, 100)
        
        # Generate request rate (correlated with CPU)
        request_rate = cpu_util * 10 + np.random.normal(0, 50, size=len(timestamps))
        request_rate = np.clip(request_rate, 0, None)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_util_percent': cpu_util,
            'mem_util_percent': memory_util,
            'request_rate': request_rate,
            'machine_id': 'synthetic_001'
        })
        
        return df
    
    def add_anomalies(self, df, anomaly_rate=0.02):
        """Add anomalies to the workload"""
        num_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'gradual'])
            
            if anomaly_type == 'spike':
                df.loc[idx:idx+3, 'cpu_util_percent'] = 95 + np.random.uniform(-5, 5)
            elif anomaly_type == 'drop':
                df.loc[idx:idx+3, 'cpu_util_percent'] = 5 + np.random.uniform(-5, 5)
            else:  # gradual
                for i in range(10):
                    if idx + i < len(df):
                        df.loc[idx+i, 'cpu_util_percent'] += i * 5
        
        return df

def main():
    """Main function to generate synthetic workload data"""
    # Create output directory if it doesn't exist
    output_dir = "data/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SyntheticWorkloadGenerator(seed=42)
    
    # Generate different workload patterns
    print("Generating synthetic workload data...")
    
    # 1. Generate 7-day workload
    print("\n1. Generating 7-day workload pattern...")
    df_7day = generator.generate_workload(duration_hours=24*7, interval_minutes=5)
    df_7day_with_anomalies = generator.add_anomalies(df_7day.copy(), anomaly_rate=0.02)
    
    # Save to CSV
    output_file = os.path.join(output_dir, "synthetic_workload_7days.csv")
    df_7day_with_anomalies.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")
    print(f"   Shape: {df_7day_with_anomalies.shape}")
    print(f"   CPU utilization - Mean: {df_7day_with_anomalies['cpu_util_percent'].mean():.2f}%, "
          f"Std: {df_7day_with_anomalies['cpu_util_percent'].std():.2f}%")
    
    # 2. Generate 24-hour workload (for quick testing)
    print("\n2. Generating 24-hour workload pattern...")
    df_24hr = generator.generate_workload(duration_hours=24, interval_minutes=5)
    df_24hr_with_anomalies = generator.add_anomalies(df_24hr.copy(), anomaly_rate=0.03)
    
    output_file = os.path.join(output_dir, "synthetic_workload_24hours.csv")
    df_24hr_with_anomalies.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")
    print(f"   Shape: {df_24hr_with_anomalies.shape}")
    
    # 3. Generate multiple machines workload
    print("\n3. Generating multi-machine workload...")
    all_machines_data = []
    
    for machine_id in range(5):  # Generate data for 5 machines
        generator_machine = SyntheticWorkloadGenerator(seed=42 + machine_id)
        df_machine = generator_machine.generate_workload(duration_hours=24*3, interval_minutes=5)
        df_machine['machine_id'] = f'synthetic_{machine_id:03d}'
        
        # Add different anomaly rates for different machines
        anomaly_rate = 0.01 + (machine_id * 0.005)  # 1% to 3%
        df_machine = generator_machine.add_anomalies(df_machine, anomaly_rate=anomaly_rate)
        
        all_machines_data.append(df_machine)
    
    df_multi = pd.concat(all_machines_data, ignore_index=True)
    output_file = os.path.join(output_dir, "synthetic_workload_multi_machine.csv")
    df_multi.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")
    print(f"   Shape: {df_multi.shape}")
    print(f"   Number of machines: {df_multi['machine_id'].nunique()}")
    
    # Display sample data
    print("\n4. Sample of generated data (first 5 rows):")
    print(df_7day_with_anomalies.head())
    
    # Generate summary statistics
    print("\n5. Summary statistics for 7-day workload:")
    print(df_7day_with_anomalies.describe())
    
    print("\n✓ Synthetic workload generation complete!")
    print(f"  All files saved to: {output_dir}/")

if __name__ == "__main__":
    main()