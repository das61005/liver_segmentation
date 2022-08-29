from click import Argument
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lung_window_Paths, soft_window_Paths, bone_window_Paths,Mask_Paths, 
                    batch_size=32, dim=(32,32,32), n_channels=1,n_classes=10, shuffle=False):
        'Initialization'
        self.lung_window_Paths = lung_window_Paths
        self.soft_window_Paths = soft_window_Paths
        self.bone_window_Paths = bone_window_Paths
        self.Mask_Paths=Mask_Paths
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.lung_window_Paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Rotate=[15,7,0,-7,-15]
        #Rotate=[0]
        # Generate data
        X, y = self.__data_generation(indexes)#

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.lung_window_Paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        index_size=len(index)
        X = np.empty((index_size, *self.dim, self.n_channels))
        y = np.empty((index_size, *self.dim, 1))#self.n_channels # y的維度跟x是一樣的
       

        # Generate data
        for i in range(index_size):
            train_list=[]
            
            ##lung_window
            #print(self.folder+'Dicom_' + ID_list[j])
            train_data = np.load(self.lung_window_Paths[index[i]])/255.0
            train_Img = Image.fromarray(train_data)
            train_small_Img = train_Img.resize((256,256), Image.ANTIALIAS)
            train_data = np.asarray(train_small_Img)
            #plt.imshow(train_data)
            train_list.append(train_data)

            ##soft_window
            train_data = np.load(self.soft_window_Paths[index[i]])/255.0
            train_Img = Image.fromarray(train_data)
            train_small_Img = train_Img.resize((256,256), Image.ANTIALIAS)
            train_data = np.asarray(train_small_Img)
            train_list.append(train_data)
            
            ##bone_window
            train_data = np.load(self.bone_window_Paths[index[i]])/255.0
            train_Img = Image.fromarray(train_data)
            train_small_Img = train_Img.resize((256,256), Image.ANTIALIAS)
            train_data = np.asarray(train_small_Img)
            train_list.append(train_data)

            train_list=np.array(train_list)
            X[i] = np.expand_dims(np.moveaxis(train_list, 0, -1),axis=0)
            

            ##Mask
            #print(self.folder+'MaskA_' +ID_list[i+1]+"\n")
            test_data = np.load(self.Mask_Paths[index[i]])
            test_Img = Image.fromarray((test_data).astype(np.uint8))#
            test_small_Img = test_Img.resize((256,256), Image.ANTIALIAS)
            test_data = np.array(test_small_Img)
            #plt.imshow(test_data)
            
            y[i] = np.expand_dims(test_data, axis=2)
        return X, y
    
    

