import tensorflow as tf
from tensorflow.keras.utils import Sequence 
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations 

# device 확인
# print(device_lib.list_local_devices())

# cpu 설정
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Augmentation

def random_transform(img):
    composition = albumentations.Compose([albumentations.HorizontalFlip(p = 0.5),
                                albumentations.VerticalFlip(p = 0.5),
                                albumentations.GridDistortion(p = 0.2),
                                albumentations.ElasticTransform(p = 0.2)])
    return composition(image=img)['image']

def augment_batch(img_batch) :
    
    for i in range(img_batch.shape[0]):
        img_batch[i] = random_transform(img_batch[i])
        
    return img_batch

 # show_batch
    
def show_batch(dataloader) : 
    train_features, train_labels = next(iter(dataloader))
    print(type(train_features))
    size = len(train_features)
    sub_size = int(size ** 0.5) +1
    plt.figure(figsize=(10, 10), dpi=80)
    for n in range(size):
        plt.subplot(sub_size, sub_size, n+1)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        plt.title(train_labels[n])
        plt.imshow(train_features[n])
    plt.show()
        
# customizing dataset

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, height, width, batch_size, shuffle=False, augment=False, mode='fit'):
        
        self.x, self.y = x_set, y_set
        self.batch_size =batch_size
        self.shuffle = shuffle 
        self.height = height
        self.width = width
        self.augment = augment
        self.mode = mode
        self.on_epoch_end()

    # 한 epoch당 shuffle 해주는 함수    
    def on_epoch_end(self):
        
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx) :
        
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        X = np.empty((self.batch_size, self.height, self.width, 1))
        
        for i, ID in enumerate(indices) :
            img = self.x[ID]
            img = img / 255.
            img = img[..., tf.newaxis]
            X[i] = img

        if self.mode == 'fit' :
            
            y = self.y[indices]
            
            if self.augment == True :
                X= augment_batch(X)
    
            return X,y

        elif self.mode == 'predict' :
            
            return X
        
        else : 
            raise AttributeError("The mode should be set to either 'fit' or 'predict'.")
    
if __name__ == '__main__' :
    
    height = 28
    width = 28
    batch_size= 64
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
   
    train_dataloader=DataGenerator(x_train, y_train, height, width, batch_size, True, augment=True)    
    test_dataloader =DataGenerator(x_train, y_train, height, width, batch_size)
    
    show_batch(train_dataloader)
    



