import random
import statistics
import math

from sklearn.model_selection import RepeatedStratifiedKFold

from tf_keras.optimizers.legacy import Adam
from tf_keras.constraints import max_norm
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Conv3D, MaxPool2D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Activation, AveragePooling2D, SpatialDropout2D
from tf_keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tf_keras.regularizers import l2
from tf_keras.utils import to_categorical
from tf_keras.backend import clear_session

from plot_generator import *
from eeg_classification.augmentation import augment_data
from eeg_classification.early_stopping import DelayedEarlyStopping


# * Assuming preprocessed data has the shape (channels, epochs, time)

def run_cnn(data_x, data_y, params):
    random_seed = random.randint(0, 4294967295) #inclucive
    random_seed = 1612385895
    print("Random Seed:", random_seed)

    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=4, random_state=random_seed)
    
    if params["num_classes"] > 2:
        data_y = to_categorical(data_y, num_classes=params["num_classes"])
    
    losses = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(rskf.split(data_x, data_y)):
        print(f"Fold {i + 1}")
        
        # Initialise the data from the current fold
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        
        # Augment the data to grow sample size
        x_train, y_train = augment_data(data_x[train_index], data_y[train_index], num_augmentations=2)
        
        # Build the model
        if params['classifier_type'] == '2CNN':
            model = generate_model_2_classes(params)
        elif params['classifier_type'] == 'EEGNet':
            model = generate_model_2_classes_eegnet(params)
        elif params['classifier_type'] == '5CNN':
            model = generate_model_5_classes(params)
        
        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        loss_function = 'categorical_crossentropy' if params["num_classes"] > 2 else 'binary_crossentropy'
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

        # Train the model
        num_epochs = 500 if params["classifier_type"] == 'EEGNet' else 25
        early_stopping = DelayedEarlyStopping(monitor='val_loss', min_epochs=5, patience=round(math.sqrt(num_epochs)), restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), batch_size=8, shuffle=False, callbacks=[early_stopping])

        # Evaluate the model
        test_loss, test_acc = model.evaluate(data_x[test_index], data_y[test_index])
        test_acc = round(test_acc * 100)
        print(f"Test Accuracy: {test_acc}%, Test Loss: {test_loss}")
        losses.append(test_loss)
        accuracies.append(test_acc)
        
        # Free up memory
        del model
        clear_session()

    print(f"Overall Accuracy: {statistics.fmean(accuracies)}%, Overall Loss: {statistics.fmean(losses)}")
    print(accuracies)
    plot_cnn(accuracies, random_seed)


def generate_model_2_classes(params):
    print("Building 2CNN")
    # Build the model 2 classes 
    model = Sequential([
            Conv3D(32, (3, 3, 3), use_bias=False, activation='relu', input_shape=(params["num_channels"], params["num_epochs"], params["num_time_points"], 1), padding='same'),
            BatchNormalization(),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.5),
            
            Flatten(),
            Dense(64, use_bias=False, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
    ])
    return model


def generate_model_2_classes_eegnet(params):
    print("Building EEGNet")
    C = params["num_channels"]
    T = params["num_time_points"] * params["num_epochs"]
    kern_length = 32 # Half of the sample rate
    dropout_rate = 0.5 # Dropout rate, affects overfitting
    F1 = 8 # Number of Temporal Filters
    D = 2 # Number of Spatial Filters
    F2 = D * F1 # Number of Pointwise Filters
    
    # Build the model 2 classes 
    model = Sequential([
            Conv2D(F1, (1, kern_length), use_bias=False, input_shape=(C, T, 1), padding='same'),
            BatchNormalization(),
            DepthwiseConv2D((C, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.)),
            BatchNormalization(),
            Activation('elu'),
            AveragePooling2D(pool_size=(1, 4)),
            SpatialDropout2D(dropout_rate),
            
            SeparableConv2D(F2, (1, 16), use_bias=False, depthwise_constraint=max_norm(1.), pointwise_constraint=max_norm(1.), padding='same'),
            BatchNormalization(),
            Activation('elu'),
            AveragePooling2D(pool_size=(1, 8)),
            SpatialDropout2D(dropout_rate),
            Flatten(),
            Dense(1, activation='sigmoid')
    ])
    return model
  

def generate_model_5_classes(params):
    print("Building 5CNN")
    # Build the model
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(params["num_channels"], params["num_epochs"], params["num_time_points"], 1)),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.2),
        Conv3D(32, (3, 3, 3), activation='relu'),  # Adding another Conv3D layer
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(params["num_classes"], activation='softmax')  # Output layer for 4 classes
    ])
    return model
