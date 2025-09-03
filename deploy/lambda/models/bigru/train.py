import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models.bigru.model import LightweightBiGRU
import os

class BiGRUTrainer:
    def __init__(self, data_path, sequence_length=24, prediction_steps=12):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, data):
        """Prepare sequences for time series prediction"""
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_steps + 1):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_steps, 0])
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the BiGRU model"""
        # Load data
        df = pd.read_csv(self.data_path)
        features = df[['cpu_util_percent', 'mem_util_percent', 'net_out']].values
        
        # Prepare sequences
        X, y = self.prepare_sequences(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Create and train model
        model = LightweightBiGRU(
            input_shape=(self.sequence_length, features.shape[1]),
            prediction_steps=self.prediction_steps
        )
        
        history = model.train(X_train, y_train, epochs=50, batch_size=32)
        
        # Evaluate
        test_loss = model.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss}")
        
        # Save model
        model.model.save('models/bigru/weights/model.h5')
        
        # Quantize and save
        size_mb = model.quantize_model()
        print(f"Quantized model size: {size_mb:.2f} MB")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, 'models/bigru/weights/scaler.pkl')
        
        return model, history

if __name__ == "__main__":
    trainer = BiGRUTrainer('F:/NCI_2025/Vineeth/load-balancer-ml/data/alibaba_subset/processed_data.csv')
    model, history = trainer.train_model()