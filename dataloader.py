import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split

composition = A.Compose([A.HorizontalFlip(p = 0.5),
                        A.VerticalFlip(p = 0.5),
                        A.GridDistortion(p = 0.2),
                        A.ElasticTransform(p = 0.2)])

def aug_fn(image, is_train=True):

    if is_train :
        data = {"image":image}
        aug_data = composition(**data)
        image = aug_data["image"]
    
    image = tf.cast(image/255.0, tf.float32)
    image = tf.expand_dims(image, -1) 
  
    return image

def process_data(image, label, is_train=False):

    # 파이썬 함수 func를 tensorflow 함수의 연산으로 래핑
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, is_train], Tout=tf.float32)
    return aug_img, label

def prepare_for_training(ds, batch_size=32, cache=True, shuffle_buffer_size=1000, is_train=False):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if is_train :
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

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
    

mnist = tf.keras.datasets.mnist

def load_data(mnist):
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).map(lambda x, y: process_data(x, y, is_train=True))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val,y_val)).map(lambda x, y: process_data(x, y, is_train=False))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).map(lambda x, y: process_data(x, y, is_train=False))
    
    train_ds = prepare_for_training(train_ds, batch_size=32, is_train=True)
    val_ds = prepare_for_training(val_ds, batch_size=32, is_train=False)
    test_ds = prepare_for_training(test_ds, batch_size=32, is_train=False)
    
    size_list = [len(x_train), len(x_val), len(x_test)]
    return train_ds, val_ds, test_ds, size_list

if __name__ == "__main__" :
    train_ds, test_ds, val_ds =load_data(mnist)
    show_batch(train_ds)
    show_batch(val_ds)
    show_batch(test_ds)
    
    
    

