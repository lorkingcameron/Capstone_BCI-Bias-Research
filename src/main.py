import scipy.io
import numpy as np
import os

PATH = os.path.dirname(os.path.dirname(__file__))


def data_preprocessing_2_classes():
    # Epoch * Channel * Time
    file_path = PATH + '/Processed Data/2 classes/'
    file_list_length = 20
    all_data = {}
    for i in range(1, file_list_length + 1):
        temp_pos_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_positive.set.mat')
        temp_pos_data = np.transpose(temp_pos_mat['data'], [1,2,0])
        all_data['S' + str(i) + '_positive'] = temp_pos_data
        
        temp_neg_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_negative.set.mat')
        temp_neg_data = np.transpose(temp_neg_mat['data'], [1,2,0])
        all_data['S' + str(i) + '_negative'] = temp_neg_data
    
    return all_data


def data_preprocessing_5_classes():
    # Epoch * Channel * Time
    file_path = PATH + '/Processed Data/5 classes/'
    file_list_length = 20
    num_classes = 4
    all_data = {}
    for i in range(1, file_list_length + 1):
        for j in  range(1, num_classes + 1):
            temp_mat = scipy.io.loadmat(file_path + 'S' + str(i) + '_class_' + str(j) + '.set.mat')
            temp_data = np.transpose(temp_mat['data'], [1,2,0])
            all_data['S' + str(i) + '_class_' + str(j)] = temp_data
    
    return all_data


def main():
    all_data = data_preprocessing_2_classes()
    print(all_data)


if __name__ == "__main__":
    main()
