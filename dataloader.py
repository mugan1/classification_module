import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from functools import partial

# 방법1
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

composition = A.Compose([A.HorizontalFlip(p = 0.5),
                        A.VerticalFlip(p = 0.5),
                        A.GridDistortion(p = 0.2),
                        A.ElasticTransform(p = 0.2)])

def aug_fn(image):

    data = {"image":image}
    aug_data = composition(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.expand_dims(aug_img, -1) 

    return aug_img

def process_data(image, label):

    # 파이썬 함수 func를 tensorflow 함수의 연산으로 래핑
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
    return aug_img, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).map(process_data)
test_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))

def prepare_for_training(ds, batch_size=32, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

train_ds = prepare_for_training(train_ds, batch_size=32)
test_ds = prepare_for_training(test_ds, batch_size=32)


def show_batch(dataloader) : 
    for img, label in dataloader.take(1):
        size = len(img)
        sub_size = int(size ** 0.5) +1
        plt.figure(figsize=(10, 10), dpi=80)
        for n in range(size):
            plt.subplot(sub_size, sub_size, n+1)
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
            plt.title(label.numpy()[n])
            plt.imshow(img[n])
    plt.show()
    
show_batch(train_ds)


