
import os
from preprocessing import *
from logistic_regression import *
from sklearn.datasets import load_iris

PATH = os.path.dirname(os.path.dirname(__file__))


def main():
    all_data, data_x, data_y = data_preprocessing_2_classes(PATH)
    # data_x = np.array(data_x)
    # print(data_x.shape)
    # print(data_y)
    
    # Load the Iris dataset
    iris = load_iris()
    data_x = iris.data
    data_y = iris.target
    # Keep the first two classes (Setosa and Versicolor)
    data_x = data_x[:100]
    data_y = data_y[:100]

    run_logistic_regression(data_x, data_y)


if __name__ == "__main__":
    main()
