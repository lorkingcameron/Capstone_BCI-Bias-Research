
import os
from preprocessing import *

PATH = os.path.dirname(os.path.dirname(__file__))


def main():
    all_data = data_preprocessing_2_classes(PATH)
    print(all_data)


if __name__ == "__main__":
    main()
