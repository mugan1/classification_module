import tensorflow as tf

class CNN(tf.keras.Model):
  def __init__(self, num_classes):
    super(CNN, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(kernel_size=(3,3), filters=16, activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, activation='relu')
    self.conv3 = tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu')
    self.pool = tf.keras.layers.MaxPooling2D((2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(32, activation='relu')
    self.d2 = tf.keras.layers.Dense(num_classes, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.pool(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

if __name__ == '__main__':
    model = CNN(num_classes=10)
    # model.build(input_shape=(None, 224, 224, 3))
    # print(model.summary())

