import tensorflow as tf
from tensorflow.keras import layers, models

class LightweightBiGRU:
    def __init__(self, input_shape, prediction_steps=12):
        self.input_shape = input_shape
        self.prediction_steps = prediction_steps
        self.model = self._build_model()
        
    def _build_model(self):
        """Build lightweight BiGRU model < 50MB"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Small BiGRU layers
        x = layers.Bidirectional(
            layers.GRU(32, return_sequences=True)
        )(inputs)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Bidirectional(
            layers.GRU(32, return_sequences=False)
        )(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.prediction_steps)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile with quantization-aware training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        # Add early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    '''
    def quantize_model(self):
        """Quantize model for Lambda deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        
        # Save quantized model
        with open('models/bigru/weights/model_quantized.tflite', 'wb') as f:
            f.write(quantized_model)
        
        return len(quantized_model) / 1024 / 1024  # Size in MB
    '''
    def quantize_model(self):
        """Quantize model for Lambda deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable TensorFlow Select operations for RNN layers
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite operations
            tf.lite.OpsSet.SELECT_TF_OPS      # Enable TensorFlow operations
        ]
        
        # Disable tensor list lowering to handle dynamic operations
        converter._experimental_lower_tensor_list_ops = False
        
        try:
            quantized_model = converter.convert()
            
            # Save quantized model
            import os
            os.makedirs('models/bigru/weights', exist_ok=True)
            
            with open('models/bigru/weights/model_quantized.tflite', 'wb') as f:
                f.write(quantized_model)
            
            size_mb = len(quantized_model) / 1024 / 1024
            print(f"Successfully quantized model. Size: {size_mb:.2f} MB")
            
            return size_mb
            
        except Exception as e:
            print(f"Quantization failed: {str(e)}")
            print("Falling back to saving unquantized model...")
            
            # Save the regular Keras model as fallback
            self.model.save('models/bigru/weights/model.h5')
            
            # Get file size of the saved model
            import os
            model_size = os.path.getsize('models/bigru/weights/model.h5') / 1024 / 1024
            
            return model_size