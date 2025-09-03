import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from model import LightweightBiGRU
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BiGRUTrainer:
    def __init__(self, data_path, sequence_length=24, prediction_steps=12):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.scaler = MinMaxScaler()
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': [],
            'prediction_samples': []
        }
        self.training_start_time = None
        self.training_end_time = None
        
    def prepare_sequences(self, data):
        """Prepare sequences for time series prediction"""
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_steps + 1):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_steps, 0])
        
        return np.array(X), np.array(y)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        # Calculate directional accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean((y_true_diff * y_pred_diff) >= 0) * 100
        else:
            directional_accuracy = 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_training_history(self, history):
        """Create comprehensive training history plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('BiGRU Model Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss over epochs
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MAE over epochs
        axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate schedule (if available)
        if 'lr' in history.history:
            axes[0, 2].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='orange')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Learning Rate: 0.001 (constant)', 
                          ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Learning Rate Info')
        
        # Plot 4: Loss improvement rate
        loss_improvement = np.diff(history.history['loss'])
        val_loss_improvement = np.diff(history.history['val_loss'])
        axes[1, 0].plot(loss_improvement, label='Training Loss Change', alpha=0.7)
        axes[1, 0].plot(val_loss_improvement, label='Validation Loss Change', alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Change')
        axes[1, 0].set_title('Loss Improvement per Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Overfitting detection
        overfitting_gap = np.array(history.history['val_loss']) - np.array(history.history['loss'])
        axes[1, 1].plot(overfitting_gap, linewidth=2, color='red')
        axes[1, 1].fill_between(range(len(overfitting_gap)), 0, overfitting_gap, 
                               where=(overfitting_gap > 0), alpha=0.3, color='red', label='Overfitting')
        axes[1, 1].fill_between(range(len(overfitting_gap)), 0, overfitting_gap, 
                               where=(overfitting_gap <= 0), alpha=0.3, color='green', label='Good Fit')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Detection')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Training statistics
        stats_text = f"""
        Final Training Loss: {history.history['loss'][-1]:.4f}
        Final Validation Loss: {history.history['val_loss'][-1]:.4f}
        Final Training MAE: {history.history['mae'][-1]:.4f}
        Final Validation MAE: {history.history['val_mae'][-1]:.4f}
        Total Epochs: {len(history.history['loss'])}
        Best Val Loss: {min(history.history['val_loss']):.4f}
        Best Val Loss Epoch: {np.argmin(history.history['val_loss']) + 1}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Training Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('models/bigru/plots', exist_ok=True)
        plt.savefig('models/bigru/plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_predictions(self, X_test, y_test, y_pred, num_samples=5):
        """Plot actual vs predicted values for multiple samples"""
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
        fig.suptitle('BiGRU Predictions vs Actual Values', fontsize=16, fontweight='bold')
        
        # Select random samples
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        for idx, sample_idx in enumerate(indices):
            # Plot predictions
            axes[idx, 0].plot(range(self.prediction_steps), y_test[sample_idx], 
                            'o-', label='Actual', linewidth=2, markersize=6)
            axes[idx, 0].plot(range(self.prediction_steps), y_pred[sample_idx], 
                            's-', label='Predicted', linewidth=2, markersize=6)
            axes[idx, 0].set_xlabel('Time Steps')
            axes[idx, 0].set_ylabel('CPU Utilization (Normalized)')
            axes[idx, 0].set_title(f'Sample {sample_idx}: Multi-step Prediction')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot error distribution
            errors = y_test[sample_idx] - y_pred[sample_idx]
            axes[idx, 1].bar(range(self.prediction_steps), errors, alpha=0.7)
            axes[idx, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[idx, 1].set_xlabel('Time Steps')
            axes[idx, 1].set_ylabel('Prediction Error')
            axes[idx, 1].set_title(f'Sample {sample_idx}: Prediction Errors')
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/bigru/plots/predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_error_analysis(self, y_test, y_pred):
        """Comprehensive error analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('BiGRU Model Error Analysis', fontsize=16, fontweight='bold')
        
        # Flatten predictions for overall analysis
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        errors = y_test_flat - y_pred_flat
        
        # Plot 1: Scatter plot of predictions vs actual
        axes[0, 0].scatter(y_test_flat, y_pred_flat, alpha=0.5, s=10)
        axes[0, 0].plot([y_test_flat.min(), y_test_flat.max()], 
                       [y_test_flat.min(), y_test_flat.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error histogram
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Zero Error')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Error Distribution (μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot (Normality Test)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Error vs prediction step
        error_by_step = []
        for step in range(self.prediction_steps):
            step_errors = y_test[:, step] - y_pred[:, step]
            error_by_step.append(step_errors)
        
        axes[1, 0].boxplot(error_by_step)
        axes[1, 0].set_xlabel('Prediction Step')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Error Distribution by Prediction Step')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cumulative error
        cumulative_abs_error = np.cumsum(np.abs(errors))
        axes[1, 1].plot(cumulative_abs_error, linewidth=2)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Cumulative Absolute Error')
        axes[1, 1].set_title('Cumulative Absolute Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Error metrics by prediction horizon
        metrics_by_step = []
        for step in range(self.prediction_steps):
            step_metrics = self.calculate_metrics(y_test[:, step], y_pred[:, step])
            metrics_by_step.append(step_metrics)
        
        metrics_df = pd.DataFrame(metrics_by_step)
        axes[1, 2].plot(range(self.prediction_steps), metrics_df['mae'], 'o-', label='MAE')
        axes[1, 2].plot(range(self.prediction_steps), metrics_df['rmse'], 's-', label='RMSE')
        axes[1, 2].set_xlabel('Prediction Step')
        axes[1, 2].set_ylabel('Error Metric')
        axes[1, 2].set_title('Error Metrics by Prediction Horizon')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/bigru/plots/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_metrics(self, metrics, history, model_size):
        """Save all metrics to JSON file"""
        metrics_data = {
            'training_info': {
                'sequence_length': self.sequence_length,
                'prediction_steps': self.prediction_steps,
                'training_duration': (self.training_end_time - self.training_start_time) if self.training_start_time else None,
                'total_epochs': len(history.history['loss']),
                'model_size_mb': model_size
            },
            'final_metrics': metrics,
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            },
            'best_performance': {
                'best_val_loss': float(min(history.history['val_loss'])),
                'best_val_loss_epoch': int(np.argmin(history.history['val_loss']) + 1),
                'best_val_mae': float(min(history.history['val_mae'])),
                'best_val_mae_epoch': int(np.argmin(history.history['val_mae']) + 1)
            }
        }
        
        os.makedirs('models/bigru/metrics', exist_ok=True)
        with open('models/bigru/metrics/training_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2, default=float)
        
        print("\n" + "="*50)
        print("METRICS SAVED TO: models/bigru/metrics/training_metrics.json")
        print("="*50)
    
    def train_model(self):
        """Train the BiGRU model with enhanced metrics and visualization"""
        print("\n" + "="*50)
        print("STARTING BIGRU TRAINING")
        print("="*50)
        
        self.training_start_time = time.time()
        
        # Load data
        print("\n📊 Loading data...")
        df = pd.read_csv(self.data_path)
        features = df[['cpu_util_percent', 'mem_util_percent', 'net_out']].values
        print(f"Data shape: {features.shape}")
        print(f"Data statistics:\n{pd.DataFrame(features, columns=['CPU', 'Memory', 'Network']).describe()}")
        
        # Prepare sequences
        print("\n🔄 Preparing sequences...")
        X, y = self.prepare_sequences(features)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        print("\n✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create and train model
        print("\n🧠 Creating BiGRU model...")
        model = LightweightBiGRU(
            input_shape=(self.sequence_length, features.shape[1]),
            prediction_steps=self.prediction_steps
        )
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/bigru/weights/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n🚀 Training model...")
        history = model.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_end_time = time.time()
        training_duration = self.training_end_time - self.training_start_time
        
        print(f"\n⏱️ Training completed in {training_duration:.2f} seconds")
        
        # Evaluate on test set
        print("\n📈 Evaluating on test set...")
        test_loss = model.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss}")
        
        # Generate predictions
        print("\n🔮 Generating predictions...")
        y_pred = model.model.predict(X_test, verbose=0)
        
        # Calculate comprehensive metrics
        print("\n📊 Calculating metrics...")
        metrics = self.calculate_metrics(y_test.flatten(), y_pred.flatten())
        
        print("\n" + "="*50)
        print("TEST SET PERFORMANCE METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.upper():20s}: {value:.4f}")
        
        # Save model
        print("\n💾 Saving model...")
        model.model.save('models/bigru/weights/model.h5')
        
        # Quantize and save
        size_mb = model.quantize_model()
        print(f"✅ Quantized model size: {size_mb:.2f} MB")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, 'models/bigru/weights/scaler.pkl')
        print("✅ Scaler saved")
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        self.plot_training_history(history)
        self.plot_predictions(X_test, y_test, y_pred)
        self.plot_error_analysis(y_test, y_pred)
        
        # Save metrics
        self.save_metrics(metrics, history, size_mb)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"✅ Model trained successfully")
        print(f"✅ Training duration: {training_duration:.2f} seconds")
        print(f"✅ Model size: {size_mb:.2f} MB")
        print(f"✅ Test RMSE: {metrics['rmse']:.4f}")
        print(f"✅ Test R²: {metrics['r2']:.4f}")
        print(f"✅ Plots saved to: models/bigru/plots/")
        print(f"✅ Metrics saved to: models/bigru/metrics/")
        print("="*50)
        
        return model, history

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    trainer = BiGRUTrainer('F:/NCI_2025/Vineeth/load-balancer-ml/data/alibaba_subset/processed_data.csv')
    model, history = trainer.train_model()