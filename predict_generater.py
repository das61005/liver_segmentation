from click import Argument
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lung_window, soft_window, bone_window, 
                    batch_size=32, dim=(32,32,32), n_channels=1,n_classes=10, shuffle=False):
        'Initialization'
        self.lung_window = lung_window
        self.soft_window = soft_window
        self.bone_window = bone_window
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.lung_window.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Rotate=[15,7,0,-7,-15]
        #Rotate=[0]
        # Generate data
        X = self.__data_generation(indexes)#

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.lung_window.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        index_size=len(index)
        X = np.empty((index_size,*self.dim, self.n_channels))
        
        # Generate data
        for i in range(index_size):
            test_list=[]

            test_Img = Image.fromarray((self.lung_window[index[i]]).astype(np.uint8))
            test_small_Img = test_Img.resize((256,256), Image.ANTIALIAS)
            test_data = np.asarray(test_small_Img)
            test_list.append(test_data)

            test_Img = Image.fromarray((self.soft_window[index[i]]).astype(np.uint8))
            test_small_Img = test_Img.resize((256,256), Image.ANTIALIAS)
            test_data = np.asarray(test_small_Img)
            test_list.append(test_data)

            test_Img = Image.fromarray((self.bone_window[index[i]]).astype(np.uint8))
            test_small_Img = test_Img.resize((256,256), Image.ANTIALIAS)
            test_data = np.asarray(test_small_Img)
            test_list.append(test_data)

            test_list=np.array(test_list)
            X[i]=np.moveaxis(test_list, 0, -1)

        return X
    
    

