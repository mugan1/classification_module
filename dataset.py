import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# resizing
x_train, x_test = x_train / 255.0, x_test / 255.0
# # Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test  = x_test[..., tf.newaxis]

def mnist_loader() :

    # Use tf.data to batch and shuffle the dataset
    
    train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(32)

    return train_ds, test_ds

def load_label():
    class_names = []
    for label in y_train :
        class_names.append(label)
    return class_names

def show_batch(image_batch, label_batch):
    size = len(image_batch)
    sub_size = int(size ** 0.5) +1

    plt.figure(figsize=(10, 10), dpi=80)
    for n in range(size):
        plt.subplot(sub_size, sub_size, n+1)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        plt.title(label_batch.numpy()[n])
        plt.imshow(image_batch[n])
    plt.show()

train_ds, test_ds = mnist_loader()
class_names = load_label()

# test
for image, label in train_ds.take(1) :
    show_batch(image, label)

