import random
import statistics
import numpy as np
import tensorflow as tf
from tf_keras.optimizers.legacy import Adam
from tf_keras.models import Sequential
from tf_keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tf_keras.callbacks import ReduceLROnPlateau
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from eeg_classification.augmentation import augment_data
import matplotlib.pyplot as plt


# Assuming your data has the shape (epochs, channels, time)
# For example:
# X_train.shape = (num_epochs, 30, 375)
# y_train.shape = (num_epochs,)

def run_cnn(data_x, data_y, params):
    random_seed = random.randint(0, 4294967295) #inclucive
    random_seed = 1612385895
    print("Random Seed:", random_seed)
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=random_seed)
    
    data_y = to_categorical(data_y, num_classes=params["num_classes"])
    
    # ! DEPRECATED - no cross-validation split
    # // x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    # // x_train, y_train = augment_data(x_train, y_train, num_augmentations=3)
    
    # // Convert labels to categorical
    # // y_train = to_categorical(y_train, num_classes=params["num_classes"])
    # // y_test = to_categorical(y_test, num_classes=params["num_classes"])
    
    # // if params["num_classes"] == 4:
    # //     model = generate_model_5_classes(params)
    # // elif params["num_classes"] == 2:
    # //     model = generate_model_2_classes(params)
    
    losses = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(sss.split(data_x, data_y)):
        # Build the model
        # model = generate_model_2_classes(params)
        model = generate_model_5_classes(params)

        # Compile the model
        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(data_x[train_index], data_y[train_index], epochs=20, validation_data=(data_x[test_index], data_y[test_index]))

        # Evaluate the model
        test_loss, test_acc = model.evaluate(data_x[test_index], data_y[test_index])
        test_acc = round(test_acc * 100)
        print(f"Test Accuracy: {test_acc}%, Test Loss: {test_loss}")
        losses.append(test_loss)
        accuracies.append(test_acc)
    
    folds = [(x + 1) for x in range(len(accuracies))]

    plt.plot(folds, accuracies, marker='.')
    plt.suptitle('CNN Model Accuracy over Folds')
    plt.title(f'Random Seed: {random_seed}', fontsize=6)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Folds')
    plt.xticks(range(min(folds), max(folds)+1))
    plt.show()
    # Make sure to close the plt object once done
    plt.close()
    print(f"Overall Accuracy: {statistics.fmean(accuracies)}%, Overall Loss: {statistics.fmean(losses)}")
    print(accuracies)


def generate_model_2_classes(params):
    # Build the model 2 classes 
    model = Sequential([
            Conv3D(32, (3, 3, 3), activation='relu', input_shape=(params["num_epochs"], params["num_channels"], params["num_time_points"], 1)),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.3),
            Conv3D(64, (3, 3, 3), activation='relu'), # Adding another Conv3D layer
            MaxPooling3D(pool_size=(2, 2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(params["num_classes"], activation='softmax')
    ])
    return model
  

# ! DEPRECATED - 5 classes model  
def generate_model_5_classes(params):
    # Build the model
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(params["num_epochs"], params["num_channels"], params["num_time_points"], 1)),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.2),
        Conv3D(64, (3, 3, 3), activation='relu'),  # Adding another Conv3D layer
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(params["num_classes"], activation='softmax')  # Output layer for 4 classes
    ])
    return model
