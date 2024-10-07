import pandas as pd
import numpy as np


class DataLoader:
    '''
    Define the DataLoader class that reads the data from the csv file and splits it into training and development sets
    '''
    # Initialize the DataLoader class
    def __init__(self, path):
        my_list = []
        for chunk in pd.read_csv(path, chunksize=2000):
            my_list.append(chunk)
        
        self.data = pd.cponcat(my_list, axis=0)
        self.data['label'] = self.data['label'].astype('float32')
        self.data = self.data.astype('float32')
        del my_list
        self.data = np.array(self.data)
    # Split the data into training and development sets
    def split_data(self):
        m, n = self.data.shape
        data_dev = self.data[0:1000].T
        Y_dev = data_dev[0]
        X_dev = data_dev[1:n]
        
        data_train = self.data[1000:m].T
        Y_train = data_train[0]
        X_train = data_train[1:n]
        
        return X_train, Y_train, X_dev, Y_dev
