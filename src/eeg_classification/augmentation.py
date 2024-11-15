import numpy as np


# Function to add Gaussian noise
def add_gaussian_noise(data, mean=0.0, std=0.05):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

# Function to flip data
def flip_data(data):
    return np.flip(data, axis=1)  # Flip along the channels axis

# Applying data augmentation
def augment_data(x_data, y_data, num_augmentations=2):
    x_augmented = []
    y_augmented = []
    for _ in range(num_augmentations):
        for i, x in enumerate(x_data):
            for j, c in enumerate(x):
                if np.random.rand() < 0.5:
                    x[j] = add_gaussian_noise(c)
            x_augmented.append(x)
            y_augmented.append(y_data[i])
    
    # Combine original and augmented data
    x_augmented = np.concatenate([x_data, np.array(x_augmented)], axis=0)
    
    # Assuming y_train is a 1D array of labels with shape (original_num_samples,)
    y_augmented = np.concatenate([y_data, y_augmented])
    
    return x_augmented, y_augmented