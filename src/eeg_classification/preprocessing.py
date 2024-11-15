import os
import scipy.io
import numpy as np


PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def extract_data(classes, classifier_type):
    file_list_length = 20
    data_x = []
    data_y = []
    
    for i in range(1, file_list_length + 1):
        # Two Class data
        if classifier_type in ['EEGNet', '2CNN']:
                temp_pos_mat = scipy.io.loadmat(f'{PATH}/Processed Data/2 classes/' + 'S' + str(i) + '_positive.set.mat')
                data_x.append(temp_pos_mat['data'])
                data_y.append(1)
                
                temp_neg_mat = scipy.io.loadmat(f'{PATH}/Processed Data/2 classes/' + 'S' + str(i) + '_negative.set.mat')
                data_x.append(temp_neg_mat['data'])
                data_y.append(0)
        # Five Class data
        else:
            for class_index, class_name in enumerate(classes):
                temp_mat = scipy.io.loadmat(f'{PATH}/Processed Data/5 classes/' + 'S' + str(i) + '_class_' + class_name + '.set.mat')
                data_x.append(temp_mat['data'])
                data_y.append(class_index)
                
    return data_x, data_y


def preprocess_data(classifier_type):
    # Get the data from the .mat files
    classes = ['0', '1'] if classifier_type in ['EEGNet', '2CNN'] else ['1', '2', '4', '5']
    data_x, data_y = extract_data(classes, classifier_type) # EEG Data Stucture: Epochs * Channels * Time
    
    # Calculate the maximum size of the dimensions
    max_epochs = max([x.shape[0] for x in data_x])
    max_channels = max([x.shape[1] for x in data_x])
    max_time = max([x.shape[2] for x in data_x])
    
    # Pad each sample to have the same number of each dimension
    data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
    data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    print(data_x_padded.shape)
    
    # Reshape the data to be in the correct format for the classifier (Samples * Channels * Epochs * Time * Value)
    data_x_reshaped = data_x_padded.transpose((0, 2, 1, 3, 4)) # Swap the channels and epochs axes
    
    # Flatten the epochs and the time axes if required
    if classifier_type == 'EEGNet':
        data_x_reshaped = data_x_reshaped.reshape((len(data_x_reshaped), max_channels, max_epochs * max_time, 1)) # Flatten the epochs and the time axes
        
    data_x_standardised = standardise_per_channel(data_x_reshaped)
    
    print(data_x_standardised.shape)
    
    return data_x_standardised, np.array(data_y), max_epochs, max_channels, max_time, len(classes)


# def data_preprocessing_2_classes_eegnet():
#     # Epoch * Channel * Time
#     file_path = f'{PATH}/Processed Data/2 classes/'
#     file_list_length = 20
#     data_x = []
#     data_y = []
#     for i in range(1, file_list_length + 1):
#         temp_pos_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_positive.set.mat')
#         data_x.append(temp_pos_mat['data'])
#         data_y.append(1)
        
#         temp_neg_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_negative.set.mat')
#         data_x.append(temp_neg_mat['data'])
#         data_y.append(0)
        
#     max_epochs = max([x.shape[0] for x in data_x])
#     max_channels = max([x.shape[1] for x in data_x])
#     max_time = max([x.shape[2] for x in data_x])
    
#     # Pad each sample to have the same number of each dimension
#     data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
#     data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
#     print(data_x_padded.shape)
    
#     print(data_x_padded[0][0][0][0])
    
#     data_x_reshaped = data_x_padded.transpose((0, 2, 1, 3, 4)) # Swap the channels and epochs axes
#     data_x_reshaped = data_x_reshaped.reshape((len(data_x_reshaped), max_channels, max_epochs * max_time, 1)) # Flatten the epochs and the time axes
    
#     print(data_x_reshaped.shape)
    
#     return data_x_reshaped, data_y, max_epochs, max_channels, max_time


# def data_preprocessing_2_classes():
#     # Epoch * Channel * Time
#     file_path = f'{PATH}/Processed Data/2 classes/'
#     file_list_length = 20
#     data_x = []
#     data_y = []
#     for i in range(1, file_list_length + 1):
#         temp_pos_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_positive.set.mat')
#         # temp_pos_data = np.transpose(temp_pos_mat['data'], [1,2,0])
#         data_x.append(temp_pos_mat['data'])
#         data_y.append(1)
        
#         temp_neg_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_negative.set.mat')
#         data_x.append(temp_neg_mat['data'])
#         data_y.append(0)
#         # temp_neg_data = np.transpose(temp_neg_mat['data'], [1,2,0])
        
#     max_epochs = max([x.shape[0] for x in data_x])
#     max_channels = max([x.shape[1] for x in data_x])
#     max_time = max([x.shape[2] for x in data_x])
    
#     # Pad each sample to have the same number of each dimension
#     data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
#     data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
#     data_x_reshaped = data_x_padded.transpose((0, 2, 1, 3, 4)) # Swap the channels and epochs axes
    
#     print(data_x_reshaped.shape)
    
#     return data_x_reshaped, data_y, max_epochs, max_channels, max_time


# def transform_5_classes_to_2_classes():
#     # Epoch * Channel * Time
#     file_path = f'{PATH}/Processed Data/5 classes/'
#     file_list_length = 20
#     classes = ['1', '2', '4', '5']
#     data_x = []
#     data_y = []
    
#     for i in range(1, file_list_length + 1):
#         for class_index, class_name in enumerate(classes):
#             temp_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_class_' + class_name + '.set.mat')
#             data_x.append(temp_mat['data'])
#             data_y.append(class_index)
        
#     max_epochs = max([x.shape[0] for x in data_x])
#     max_channels = max([x.shape[1] for x in data_x])
#     max_time = max([x.shape[2] for x in data_x])
    
#     # Pad each sample to have the same number of each dimension
#     data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
#     data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
#     data_x_reshaped = data_x_padded.transpose((0, 2, 1, 3, 4)) # Swap the channels and epochs axes
    
#     data_x_standardised = standardise_per_channel(data_x_reshaped)
        
#     return data_x_standardised, data_y, max_epochs, max_channels, max_time


# def data_preprocessing_5_classes():    
#     # Epoch * Channel * Time
#     file_path = f'{PATH}/Processed Data/5 classes/'
#     file_list_length = 20
#     classes = ['1', '2', '4', '5']
#     data_x = []
#     data_y = []
    
#     for i in range(1, file_list_length + 1):
#         for class_index, class_name in enumerate(classes):
#             temp_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_class_' + class_name + '.set.mat')
#             data_x.append(temp_mat['data'])
#             data_y.append(class_index)
        
#     max_epochs = max([x.shape[0] for x in data_x])
#     max_channels = max([x.shape[1] for x in data_x])
#     max_time = max([x.shape[2] for x in data_x])
    
#     # Pad each sample to have the same number of each dimension
#     data_x_padded = [np.pad(x, ((0, max_epochs - x.shape[0]), (0, max_channels - x.shape[1]), (0, max_time - x.shape[2])), mode='constant') for x in data_x]
#     data_x_padded = np.stack(data_x_padded, axis=0)[..., np.newaxis]  # Shape: (batch_size, max_epochs, channels, time, 1)
    
#     data_x_reshaped = data_x_padded.transpose((0, 2, 1, 3, 4)) # Swap the channels and epochs axes
    
#     data_x_standardised = standardise_per_channel(data_x_reshaped)
        
#     return data_x_standardised, data_y, max_epochs, max_channels, max_time


def standardise_per_channel(data, epsilon=1e-8):
    # data shape: (samples, epochs, channels, time)
    
    # Calculate mean and std along the time axis (axis=-1) for each sample, epoch, and channel    
    channel_stds = np.std(data, axis=(2, 3), keepdims=True)    # Shape: (samples, epochs, channels, 1)
    channel_means = np.mean(data, axis=(2, 3), keepdims=True)  # Shape: (1, 1, channels, 1)
    
    # Standardize each channel independently along the time axis
    standardized_data = (data - channel_means) / (channel_stds + epsilon)
    
    return standardized_data
