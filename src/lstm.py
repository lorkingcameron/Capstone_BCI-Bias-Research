import numpy as np
import tensorflow as tf
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout, Masking, LayerNormalization, Input
from tf_keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def run_lstm(data_x, data_y, params):
    # Reshape data for LSTM input: (samples, time_steps, features)
    # Here, we can combine epochs and time into a single time dimension.
    data_x_reshaped = data_x.reshape(len(data_x), params["num_epochs"] * params["num_time_points"], params["num_channels"])
    print(data_x_reshaped.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_x_reshaped, data_y, test_size=0.4, random_state=42)
    
    # Build the model
    model = Sequential()
    model.add(Input(shape=(params["num_epochs"] * params["num_time_points"], params["num_channels"])))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LayerNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))  # Assuming 4 classes
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Reduce the learning rate when the validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
    
    # Save the entire model
    model.save("/src/model")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.2f}')
