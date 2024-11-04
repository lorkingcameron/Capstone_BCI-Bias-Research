
import os
from preprocessing import *
from cnn import *

PATH = os.path.dirname(os.path.dirname(__file__))


def main():
    # all_data, data_x, data_y, max_epochs, max_channels, time_points = data_preprocessing_2_classes(PATH)
    all_data, data_x, data_y, max_epochs, max_channels, time_points = data_preprocessing_5_classes(PATH)

    print(data_x[0].shape)
    print(data_y)
    print(max_epochs, max_channels, time_points)

    cnn_hyperparameters = {
        'num_epochs': max_epochs,
        'num_channels': max_channels,
        'num_time_points': time_points,
        'num_classes': 4
    }
    
    run_cnn(data_x, data_y, cnn_hyperparameters)
    

if __name__ == "__main__":
    main()
