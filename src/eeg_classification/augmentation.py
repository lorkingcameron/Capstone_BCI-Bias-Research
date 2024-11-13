import numpy as np


# Function to add Gaussian noise
def add_gaussian_noise(data, mean=0.0, std=0.05):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

# Function to time shift
def time_shift(data, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift, axis=2)  # Shift along the time axis

# Function to scale the amplitude
def scale_amplitude(data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale_factor

# Function to shuffle channels
def shuffle_channels(data):
    channel_indices = np.arange(data.shape[1])
    np.random.shuffle(channel_indices)
    return data[:, channel_indices, :, :]

# Function to flip data
def flip_data(data):
    return np.flip(data, axis=1)  # Flip along the channels axis

# Applying data augmentation
def augment_data(x_data, y_data, num_augmentations=3):
    augmented_data = []
    for _ in range(num_augmentations):
        for x in x_data:
            # Apply each augmentation randomly
            if np.random.rand() < 0.5:
                x = add_gaussian_noise(x)
            if np.random.rand() < 0.5:
                x = time_shift(x)
            if np.random.rand() < 0.5:
                x = scale_amplitude(x)
            if np.random.rand() < 0.5:
                x = shuffle_channels(x)
            if np.random.rand() < 0.5:
                x = flip_data(x)
            augmented_data.append(x)
    
    # Combine original and augmented data
    augmented_data = np.array(augmented_data)
    x_augmented = np.concatenate([x_data, augmented_data], axis=0)
    
    # Assuming y_train is a 1D array of labels with shape (original_num_samples,)
    y_augmented = np.repeat(y_data, num_augmentations + 1, axis=0)
    
    return x_augmented, y_augmented