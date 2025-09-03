import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

class AlibabaDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        # Define column names based on Alibaba trace format
        self.column_names = [
            'machine_id',      # Machine ID (e.g., m_1932)
            'timestamp',       # Timestamp (seconds)
            'cpu_util',        # CPU utilization (0-100)
            'mem_util',        # Memory utilization (0-100)
            'disk_io_percent', # Disk I/O percentage
            'net_in',          # Network in
            'net_out'          # Network out
        ]
        
    def detect_file_format(self, sample_size=1000):
        """Detect the file format by reading a small sample"""
        print("   Detecting file format...")
        
        # First try comma-separated without header (common for Alibaba dataset)
        try:
            sample = pd.read_csv(self.data_path, nrows=sample_size, header=None)
            if sample.shape[1] > 1:
                print(f"   Detected {sample.shape[1]} columns (comma-separated, no header)")
                
                # Show sample of first row
                print(f"   First row sample: {sample.iloc[0].values[:5]}...")
                
                # Check if first row looks like headers or data
                first_row = sample.iloc[0]
                # If first column starts with 'm_' it's likely data, not headers
                if str(first_row[0]).startswith('m_'):
                    print("   First row appears to be data (not headers)")
                
                # Adjust column names based on actual number of columns
                if sample.shape[1] == 7:
                    return self.column_names, ',', None
                elif sample.shape[1] == 6:
                    # If 6 columns, might be missing network out
                    return self.column_names[:-1], ',', None
                elif sample.shape[1] == 5:
                    # If 5 columns, might be missing both network columns
                    return self.column_names[:-2], ',', None
                elif sample.shape[1] >= 9:
                    # Might have additional columns
                    extended_names = self.column_names + [f'col_{i}' for i in range(7, sample.shape[1])]
                    return extended_names[:sample.shape[1]], ',', None
                else:
                    print(f"   Found {sample.shape[1]} columns")
                    # Create generic column names
                    return [f'col_{i}' for i in range(sample.shape[1])], ',', None
        except Exception as e:
            print(f"   Error with comma-separated (no header): {e}")
        
        # Try with header (in case it's a different format)
        try:
            sample = pd.read_csv(self.data_path, nrows=5)
            if len(sample.columns) > 1:
                # Check if columns look like actual headers or data
                if any(col.startswith('m_') for col in sample.columns):
                    print("   WARNING: File appears to have no headers but pandas read first row as headers")
                    print("   Will treat as no-header file")
                    # Re-read without headers
                    sample = pd.read_csv(self.data_path, nrows=sample_size, header=None)
                    if sample.shape[1] >= 7:
                        return self.column_names[:sample.shape[1]], ',', None
                    else:
                        return [f'col_{i}' for i in range(sample.shape[1])], ',', None
                else:
                    print(f"   Detected comma-separated format with headers")
                    print(f"   Columns: {list(sample.columns)}")
                    return list(sample.columns), ',', 0
        except:
            pass
        
        # Try space-separated
        try:
            sample = pd.read_csv(self.data_path, nrows=sample_size, header=None, sep=r'\s+')
            if sample.shape[1] > 1:
                print(f"   Detected {sample.shape[1]} columns (space-separated, no header)")
                
                if sample.shape[1] == 7:
                    return self.column_names, r'\s+', None
                elif sample.shape[1] == 6:
                    return self.column_names[:-1], r'\s+', None
                elif sample.shape[1] == 5:
                    return self.column_names[:-2], r'\s+', None
                else:
                    return [f'col_{i}' for i in range(sample.shape[1])], r'\s+', None
        except:
            pass
            
        raise ValueError("Unable to detect file format. Please check the file.")
    
    def load_subset_chunked(self, machines=20, days=7, chunksize=100000):
        """Load subset of data using chunked reading for large files"""
        # First detect the file format
        column_names, separator, header = self.detect_file_format()
        print(f"   Using columns: {column_names}")
        
        # Identify key columns
        machine_col = None
        cpu_col = None
        mem_col = None
        time_col = None
        
        for i, col in enumerate(column_names):
            if 'machine' in str(col).lower() or col == 'col_0':
                machine_col = col
            elif 'cpu' in str(col).lower() or (col == 'col_2' and len(column_names) >= 3):
                cpu_col = col
            elif 'mem' in str(col).lower() or (col == 'col_3' and len(column_names) >= 4):
                mem_col = col
            elif 'time' in str(col).lower() or (col == 'col_1' and len(column_names) >= 2):
                time_col = col
        
        if not machine_col:
            raise ValueError("Could not identify machine ID column")
        
        print(f"   Machine column: {machine_col}")
        print(f"   CPU column: {cpu_col}")
        print(f"   Time column: {time_col}")
        
        # First pass: Get unique machines and their statistics
        print("\n   First pass: Analyzing machine patterns...")
        machine_stats = self._analyze_machines_chunked(column_names, machine_col, cpu_col, chunksize, separator, header)
        
        if not machine_stats:
            raise ValueError("No machine data found in file")
        
        # Select diverse machines
        selected_machines = self._select_diverse_machines_from_stats(machine_stats, n=machines)
        print(f"\n   Selected {len(selected_machines)} diverse machines:")
        for i, m in enumerate(selected_machines[:5]):
            stats = machine_stats[m]
            print(f"     {m}: mean_cpu={stats['mean']:.1f}%, variance={stats['variance']:.1f}")
        if len(selected_machines) > 5:
            print(f"     ... and {len(selected_machines)-5} more")
        
        # Second pass: Extract data for selected machines
        print("\n   Second pass: Extracting data for selected machines...")
        df_subset = self._extract_machine_data_chunked(column_names, selected_machines, 
                                                      machine_col, time_col, days, chunksize, separator, header)
        
        return df_subset
    
    def _analyze_machines_chunked(self, column_names, machine_col, cpu_col, chunksize, separator, header):
        """Analyze machines in chunks to get statistics"""
        machine_stats = {}
        chunk_count = 0
        total_rows = 0
        
        # Read file in chunks
        if separator == ',':
            chunk_reader = pd.read_csv(self.data_path, chunksize=chunksize, header=header, 
                                     names=column_names if header is None else None, 
                                     on_bad_lines='skip')
        else:
            chunk_reader = pd.read_csv(self.data_path, chunksize=chunksize, header=None, 
                                     sep=separator, names=column_names, on_bad_lines='skip')
        
        for chunk in chunk_reader:
            chunk_count += 1
            total_rows += len(chunk)
            
            if chunk_count % 10 == 0:
                print(f"      Processed {chunk_count} chunks ({total_rows:,} rows)...")
            
            # Calculate statistics for each machine
            if cpu_col and cpu_col in chunk.columns:
                for machine_id, group in chunk.groupby(machine_col):
                    if machine_id not in machine_stats:
                        machine_stats[machine_id] = {
                            'count': 0,
                            'sum': 0,
                            'sum_sq': 0,
                            'min': float('inf'),
                            'max': float('-inf')
                        }
                    
                    stats = machine_stats[machine_id]
                    cpu_values = pd.to_numeric(group[cpu_col], errors='coerce').dropna().values
                    
                    if len(cpu_values) > 0:
                        stats['count'] += len(cpu_values)
                        stats['sum'] += cpu_values.sum()
                        stats['sum_sq'] += (cpu_values ** 2).sum()
                        stats['min'] = min(stats['min'], cpu_values.min())
                        stats['max'] = max(stats['max'], cpu_values.max())
            
            # Limit machines for memory efficiency
            if len(machine_stats) > 500:
                print(f"      Found {len(machine_stats)} machines, stopping analysis...")
                break
        
        print(f"      Total chunks: {chunk_count}, Total rows: {total_rows:,}")
        print(f"      Unique machines found: {len(machine_stats)}")
        
        # Calculate variance and mean
        machines_to_remove = []
        for machine_id, stats in machine_stats.items():
            if stats['count'] > 1:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                stats['variance'] = max(0, variance)
                stats['mean'] = mean
            else:
                machines_to_remove.append(machine_id)
        
        # Remove machines with insufficient data
        for machine_id in machines_to_remove:
            del machine_stats[machine_id]
        
        return machine_stats
    
    def _select_diverse_machines_from_stats(self, machine_stats, n=20):
        """Select machines with different workload patterns"""
        # Sort by variance
        sorted_machines = sorted(machine_stats.items(), 
                               key=lambda x: x[1]['variance'], 
                               reverse=True)
        
        n = min(n, len(sorted_machines))
        n_third = max(1, n // 3)
        
        # High variance machines
        high_var = [m[0] for m in sorted_machines[:n_third]]
        
        # Low variance machines
        low_var = [m[0] for m in sorted_machines[-n_third:]]
        
        # Medium variance machines
        remaining = len(sorted_machines) - 2 * n_third
        if remaining > 0:
            mid_start = n_third
            mid_end = mid_start + (n - 2 * n_third)
            medium_var = [m[0] for m in sorted_machines[mid_start:mid_end]]
        else:
            medium_var = []
        
        selected = high_var + medium_var + low_var
        return selected[:n]
    
    def _extract_machine_data_chunked(self, column_names, selected_machines, 
                                     machine_col, time_col, days, chunksize, separator, header):
        """Extract data for selected machines"""
        all_data = []
        chunk_count = 0
        total_rows_collected = 0
        rows_per_machine = {}
        
        # Target rows per machine (5-minute intervals for N days)
        if time_col:
            target_rows = days * 24 * 12  # Assuming 5-min intervals
        else:
            target_rows = 10000  # Default if no timestamp
        
        print(f"      Target rows per machine: {target_rows}")
        
        # Read file in chunks
        if separator == ',':
            chunk_reader = pd.read_csv(self.data_path, chunksize=chunksize, header=header,
                                     names=column_names if header is None else None,
                                     on_bad_lines='skip')
        else:
            chunk_reader = pd.read_csv(self.data_path, chunksize=chunksize, header=None,
                                     sep=separator, names=column_names, on_bad_lines='skip')
        
        for chunk in chunk_reader:
            chunk_count += 1
            
            # Filter for selected machines
            mask = chunk[machine_col].isin(selected_machines)
            chunk_filtered = chunk[mask].copy()
            
            if len(chunk_filtered) > 0:
                # Track rows per machine
                for machine_id in selected_machines:
                    machine_data = chunk_filtered[chunk_filtered[machine_col] == machine_id]
                    
                    if len(machine_data) > 0:
                        if machine_id not in rows_per_machine:
                            rows_per_machine[machine_id] = 0
                        
                        # Check if we need more data for this machine
                        if rows_per_machine[machine_id] < target_rows:
                            rows_to_take = min(len(machine_data), 
                                             target_rows - rows_per_machine[machine_id])
                            all_data.append(machine_data.iloc[:rows_to_take])
                            rows_per_machine[machine_id] += rows_to_take
                            total_rows_collected += rows_to_take
            
            if chunk_count % 10 == 0:
                print(f"      Processed {chunk_count} chunks, collected {total_rows_collected:,} rows")
                machines_complete = sum(1 for m in selected_machines 
                                      if rows_per_machine.get(m, 0) >= target_rows)
                print(f"      Machines complete: {machines_complete}/{len(selected_machines)}")
            
            # Check if we have enough data
            if all([rows_per_machine.get(m, 0) >= target_rows for m in selected_machines]):
                print(f"      ✓ Collected sufficient data from {chunk_count} chunks")
                break
            
            # Safety limit
            if total_rows_collected > 500000:
                print(f"      Reached safety limit of 500K rows")
                break
        
        if not all_data:
            raise ValueError("No data collected for selected machines")
        
        print(f"\n      Total rows collected: {total_rows_collected:,}")
        
        # Combine all data
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Process timestamps if available
        if time_col and time_col in df_combined.columns:
            print("      Processing timestamps...")
            # Convert timestamp to datetime (assuming seconds since epoch)
            df_combined['timestamp'] = pd.to_numeric(df_combined[time_col], errors='coerce')
            
            # If timestamps look like seconds since epoch
            if df_combined['timestamp'].min() > 1e9:
                df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'], unit='s')
            else:
                # Otherwise assume they're relative seconds, create synthetic timestamps
                start_time = datetime.now() - timedelta(days=days)
                df_combined['timestamp'] = start_time + pd.to_timedelta(df_combined['timestamp'], unit='s')
        else:
            print("      No timestamp column found, creating synthetic timestamps...")
            # Create synthetic timestamps
            start_time = datetime.now() - timedelta(days=days)
            
            # Group by machine and assign timestamps
            dfs_with_time = []
            for machine_id in df_combined[machine_col].unique():
                machine_df = df_combined[df_combined[machine_col] == machine_id].copy()
                machine_df['timestamp'] = pd.date_range(start=start_time, 
                                                       periods=len(machine_df), 
                                                       freq='5T')
                dfs_with_time.append(machine_df)
            
            df_combined = pd.concat(dfs_with_time, ignore_index=True)
        
        # Rename columns to standard names
        rename_dict = {machine_col: 'machine_id'}
        
        if 'cpu' in column_names:
            cpu_col_idx = column_names.index('cpu')
            rename_dict[column_names[cpu_col_idx]] = 'cpu_util_percent'
        elif len(column_names) > 2:
            rename_dict[column_names[2]] = 'cpu_util_percent'
            
        if 'mem' in column_names:
            mem_col_idx = column_names.index('mem')
            rename_dict[column_names[mem_col_idx]] = 'mem_util_percent'
        elif len(column_names) > 3:
            rename_dict[column_names[3]] = 'mem_util_percent'
        
        df_combined = df_combined.rename(columns=rename_dict)
        
        # Ensure numeric columns are numeric
        numeric_cols = ['cpu_util_percent', 'mem_util_percent', 'disk_io_percent', 'net_in', 'net_out']
        for col in numeric_cols:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
        return df_combined

def main():
    """Main function to process Alibaba cluster data"""
    input_dir = "data/alibaba_subset"
    output_dir = "data/processed"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Alibaba Cluster Data Preprocessing")
    print("=" * 50)
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in {input_dir}")
        return
    
    for csv_file in csv_files:
        print(f"\n📁 Processing: {os.path.basename(csv_file)}")
        
        # Check file size
        file_size = os.path.getsize(csv_file) / (1024**3)
        print(f"   File size: {file_size:.2f} GB")
        
        try:
            processor = AlibabaDataProcessor(csv_file)
            
            # Process the data
            print("\n   Loading and processing data...")
            df_processed = processor.load_subset_chunked(machines=20, days=7, chunksize=100000)
            
            # Display results
            print(f"\n   ✓ Data processing complete!")
            print(f"   Shape: {df_processed.shape}")
            print(f"   Columns: {list(df_processed.columns)}")
            
            if 'machine_id' in df_processed.columns:
                print(f"   Unique machines: {df_processed['machine_id'].nunique()}")
                print(f"   Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
            
            # Save processed data
            output_file = os.path.join(output_dir, f"processed_{os.path.basename(csv_file)}")
            print(f"\n   Saving to: {output_file}")
            df_processed.to_csv(output_file, index=False)
            
            # Display sample
            print("\n   Sample data:")
            print(df_processed.head(10))
            
            # Statistics
            if 'cpu_util_percent' in df_processed.columns:
                print("\n   CPU Utilization Statistics:")
                print(f"     Mean: {df_processed['cpu_util_percent'].mean():.2f}%")
                print(f"     Std:  {df_processed['cpu_util_percent'].std():.2f}%")
                print(f"     Min:  {df_processed['cpu_util_percent'].min():.2f}%")
                print(f"     Max:  {df_processed['cpu_util_percent'].max():.2f}%")
            
            memory_mb = df_processed.memory_usage(deep=True).sum() / 1024**2
            print(f"\n   Output file memory usage: {memory_mb:.2f} MB")
            
        except Exception as e:
            print(f"\n   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Processing complete!")

if __name__ == "__main__":
    main()