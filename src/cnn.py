import numpy as np
import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from augmentation import augment_data


# Assuming your data has the shape (epochs, channels, time)
# For example:
# X_train.shape = (num_epochs, 30, 375)
# y_train.shape = (num_epochs,)

def run_cnn(data_x, data_y, params):
    
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    # x_train, y_train = augment_data(x_train, y_train, num_augmentations=3)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=params["num_classes"])
    y_test = to_categorical(y_test, num_classes=params["num_classes"])

    # Build the model
    if params["num_classes"] == 4:
        model = generate_model_5_classes(params)
    elif params["num_classes"] == 2:
        model = generate_model_2_classes(params)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")


def generate_model_2_classes(params):
        # Build the model 2 classes 
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(params["num_epochs"], params["num_channels"], params["num_time_points"], 1)),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(params["num_classes"], activation='softmax')
    ])
    return model
    
def generate_model_5_classes(params):
    # Build the model
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(params["num_epochs"], params["num_channels"], params["num_time_points"], 1)),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.3),
        Conv3D(64, (3, 3, 3), activation='relu'),  # Adding another Conv3D layer
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Output layer for 4 classes
    ])
    return model
