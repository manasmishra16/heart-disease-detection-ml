"""
Advanced Deep Learning Models for Heart Disease Detection
Implements: CNN, LSTM, CNN-LSTM hybrid, Multi-Input models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class ResidualBlock(layers.Layer):
    """Residual block for deep CNN"""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.shortcut = layers.Conv1D(filters, 1, padding='same')
        
    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = layers.Dropout(0.2)(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Shortcut connection
        if inputs.shape[-1] != self.filters:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
        
        # Add and activate
        x = layers.Add()([x, shortcut])
        x = tf.nn.relu(x)
        return x


class AttentionBlock(layers.Layer):
    """Squeeze-and-Excitation attention mechanism"""
    
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        filters = input_shape[-1]
        self.gap = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(filters // self.reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, filters))
        self.multiply = layers.Multiply()
        
    def call(self, inputs):
        se = self.gap(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return self.multiply([inputs, se])


def create_deep_cnn_model(input_shape=(1000, 1), num_classes=1):
    """
    Deep Residual 1D-CNN with Attention
    Target: 94-96% accuracy on ECG signals
    """
    inputs = keras.Input(shape=input_shape, name='ecg_input')
    
    # Initial convolution
    x = layers.Conv1D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks - Stage 1
    x = ResidualBlock(64)(x)
    x = ResidualBlock(64)(x)
    
    # Downsample and increase filters
    x = layers.Conv1D(128, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks - Stage 2
    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)
    
    # Downsample and increase filters
    x = layers.Conv1D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks - Stage 3
    x = ResidualBlock(256)(x)
    x = ResidualBlock(256)(x)
    
    # Attention mechanism
    x = AttentionBlock()(x)
    
    # Dual pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Classification head
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='deep_cnn')
    return model


def create_cnn_lstm_model(input_shape=(1000, 1), num_classes=1):
    """
    CNN-LSTM Hybrid for time-series analysis
    Target: 95-97% accuracy
    """
    inputs = keras.Input(shape=input_shape, name='ecg_input')
    
    # CNN feature extraction
    x = layers.Conv1D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Bidirectional LSTM for temporal dependencies
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='cnn_lstm')
    return model


def create_lstm_model(input_shape=(1000, 1), num_classes=1):
    """
    Bidirectional LSTM for temporal pattern recognition
    """
    inputs = keras.Input(shape=input_shape, name='ecg_input')
    
    # Stacked Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='lstm')
    return model


def create_gru_model(input_shape=(1000, 1), num_classes=1):
    """
    Bidirectional GRU (faster than LSTM)
    """
    inputs = keras.Input(shape=input_shape, name='ecg_input')
    
    # Stacked Bidirectional GRU
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='gru')
    return model


def create_enhanced_mlp_model(input_shape=(13,), num_classes=1):
    """
    Enhanced MLP for clinical features
    """
    inputs = keras.Input(shape=input_shape, name='clinical_input')
    
    # Deep MLP with batch normalization
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='enhanced_mlp')
    return model


def create_multi_input_model(ecg_shape=(1000, 1), clinical_shape=(13,), num_classes=1):
    """
    Multi-Input Model: ECG + Clinical Data
    Target: 96-98% accuracy
    """
    # ECG Branch (CNN-LSTM)
    ecg_input = keras.Input(shape=ecg_shape, name='ecg_input')
    x1 = layers.Conv1D(64, 7, padding='same')(ecg_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = ResidualBlock(64)(x1)
    
    x1 = layers.Conv1D(128, 5, strides=2, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = ResidualBlock(128)(x1)
    
    x1 = layers.Bidirectional(layers.LSTM(64))(x1)
    ecg_features = layers.Dense(64, activation='relu')(x1)
    
    # Clinical Branch (MLP)
    clinical_input = keras.Input(shape=clinical_shape, name='clinical_input')
    x2 = layers.Dense(128, activation='relu')(clinical_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Dense(64, activation='relu')(x2)
    clinical_features = layers.BatchNormalization()(x2)
    
    # Fusion Layer
    combined = layers.Concatenate()([ecg_features, clinical_features])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = Model(
        inputs=[ecg_input, clinical_input],
        outputs=outputs,
        name='multi_input_model'
    )
    return model


def compile_model(model, learning_rate=0.001):
    """Compile model with appropriate settings"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model


def get_callbacks(model_path, patience=15):
    """Get training callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    return callbacks


def main():
    """Demo: Create all models and print summaries"""
    print("="*60)
    print("DEEP LEARNING MODELS")
    print("="*60)
    
    models = {
        'Deep CNN': create_deep_cnn_model(),
        'CNN-LSTM': create_cnn_lstm_model(),
        'LSTM': create_lstm_model(),
        'GRU': create_gru_model(),
        'Enhanced MLP': create_enhanced_mlp_model(),
    }
    
    for name, model in models.items():
        print(f"\n📊 {name}")
        print(f"  Parameters: {model.count_params():,}")
        print(f"  Layers: {len(model.layers)}")
    
    print("\n" + "="*60)
    print("All models created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
