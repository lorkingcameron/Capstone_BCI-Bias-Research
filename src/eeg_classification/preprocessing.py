import os
import scipy.io
import numpy as np
from sklearn.utils import shuffle


PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def data_preprocessing_2_classes():
    # Epoch * Channel * Time
    file_path = f'{PATH}/Processed Data/2 classes/'
    file_list_length = 20
    all_data = {}
    data_x = []
    data_y = []
    for i in range(1, file_list_length + 1):
        temp_pos_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_positive.set.mat')
        # temp_pos_data = np.transpose(temp_pos_mat['data'], [1,2,0])
        data_x.append(temp_pos_mat['data'])
        data_y.append(1)
        all_data['S' + str(i) + '_positive'] = temp_pos_mat['data']
        
        temp_neg_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_negative.set.mat')
        data_x.append(temp_neg_mat['data'])
        data_y.append(0)
        # temp_neg_data = np.transpose(temp_neg_mat['data'], [1,2,0])
        all_data['S' + str(i) + '_negative'] = temp_neg_mat['data']
        
    max_epochs = max([x.shape[0] for x in data_x])
    max_channels = max([x.shape[1] for x in data_x])
    max_time = max([x.shape[2] for x in data_x])
    
    # Pad each sample to have the same number of epochs (padding along the epochs axis)
    data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
    data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
    print(data_x_padded.shape)
    
    # ! DEPRECATED - no longer required due to stratified split
    # shuffle_seed = random.randint(0, 4294967295) #inclucive
    # print("Shuffle Seed:", shuffle_seed)
    # x_shuffled, y_shuffled = shuffle(np.array(data_x_padded), np.array(data_y), random_state=shuffle_seed)
    
    return all_data, data_x_padded, data_y, max_epochs, max_channels, max_time


def transform_5_classes_to_2_classes():
    # Epoch * Channel * Time
    file_path = f'{PATH}/Processed Data/5 classes/'
    file_list_length = 20
    classes = ['1', '2', '4', '5']
    all_data = {}
    data_x = []
    data_y = []
    
    for i in range(1, file_list_length + 1):
        for class_index, class_name in enumerate(classes):
            temp_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_class_' + class_name + '.set.mat')
            data_x.append(temp_mat['data'])
            data_y.append(class_index)
            
            all_data['S' + str(i) + '_class_' + class_name] = temp_mat['data']
        
    max_epochs = max([x.shape[0] for x in data_x])
    max_channels = max([x.shape[1] for x in data_x])
    max_time = max([x.shape[2] for x in data_x])
    
    # Pad each sample to have the same number of epochs (padding along the epochs axis)
    data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
    data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
    data_x_standardised = standardise_per_channel(data_x_padded)

    # ! DEPRECATED - no longer required due to stratified split
    # shuffle_seed = random.randint(0, 4294967295) #inclucive
    # print("Shuffle Seed:", shuffle_seed)
    # x_shuffled, y_shuffled = shuffle(np.array(data_x_standardised), np.array(data_y), random_state=shuffle_seed)
        
    return all_data, data_x_standardised, data_y, max_epochs, max_channels, max_time


def data_preprocessing_5_classes():    
    # Epoch * Channel * Time
    file_path = f'{PATH}/Processed Data/5 classes/'
    file_list_length = 20
    classes = ['1', '2', '4', '5']
    all_data = {}
    data_x = []
    data_y = []
    
    for i in range(1, file_list_length + 1):
        for class_index, class_name in enumerate(classes):
            temp_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_class_' + class_name + '.set.mat')
            data_x.append(temp_mat['data'])
            data_y.append(class_index)
            
            all_data['S' + str(i) + '_class_' + class_name] = temp_mat['data']
        
    max_epochs = max([x.shape[0] for x in data_x])
    max_channels = max([x.shape[1] for x in data_x])
    max_time = max([x.shape[2] for x in data_x])
    
    # Pad each sample to have the same number of epochs (padding along the epochs axis)
    data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
    data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
    # data_x_standardised = standardise_per_channel(data_x_padded)

    # ! DEPRECATED - no longer required due to stratified split
    # shuffle_seed = random.randint(0, 4294967295) #inclucive
    # print("Shuffle Seed:", shuffle_seed)
    # x_shuffled, y_shuffled = shuffle(np.array(data_x_standardised), np.array(data_y), random_state=shuffle_seed)
        
    return all_data, data_x_padded, data_y, max_epochs, max_channels, max_time


def standardise_per_channel(data, epsilon=1e-8):
    # data shape: (samples, epochs, channels, time)
    
    # Calculate mean and std along the time axis (axis=-1) for each sample, epoch, and channel    
    channel_stds = np.std(data, axis=(0, 1, 3), keepdims=True)    # Shape: (samples, epochs, channels, 1)
    channel_means = np.mean(data, axis=(0, 1, 3), keepdims=True)  # Shape: (1, 1, channels, 1)
    
    # Standardize each channel independently along the time axis
    standardized_data = (data - channel_means) / (channel_stds + epsilon)
    
    return standardized_data
