import tensorflow as tf
from models.cnn import CNN
from dataloader import load_data
import os
import numpy as np
from tensorflow.python.client import device_lib
import math
from tensorflow.keras.utils import Progbar
from tqdm import tqdm
import matplotlib.pyplot as plt

# device 확인
# print(device_lib.list_local_devices())

# cpu/gpu 설정
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Trainer 클래스 정의
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.loss_fn(y, logits)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def train(self, x_train, y_train, x_val=None, y_val=None, batch_size=32, epochs=10):
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = tf.keras.metrics.Mean()
            train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
            for i in tqdm(range(0, len(x_train), batch_size)):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                batch_loss = self.train_step(x_batch, y_batch)
                train_loss(batch_loss)
                train_acc(y_batch, self.model(x_batch))
            history['train_loss'].append(train_loss.result().numpy())
            history['train_acc'].append(train_acc.result().numpy())
            print(f"train_loss: {history['train_loss'][-1]:.4f} - train_acc: {history['train_acc'][-1]:.4f}")
            
            if x_val is not None and y_val is not None:
                val_loss = tf.keras.metrics.Mean()
                val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
                for i in tqdm(range(0, len(x_val), batch_size)):
                    x_batch = x_val[i:i+batch_size]
                    y_batch = y_val[i:i+batch_size]
                    val_loss(self.loss_fn(y_batch, self.model(x_batch)))
                    val_acc(y_batch, self.model(x_batch))
                history['val_loss'].append(val_loss.result().numpy())
                history['val_acc'].append(val_acc.result().numpy())
                print(f"val_loss: {history['val_loss'][-1]:.4f} - val_acc: {history['val_acc'][-1]:.4f}")

        return history
    
if __name__ == "__main__" :
    
    # dataset
    mnist = tf.keras.datasets.mnist
    train_ds, val_ds, test_ds, size_list = load_data(mnist)
    
        # 하이퍼파라미터 설정
    epoch = 1
    batch_size = 32
    num_classes = 10
    model = CNN(num_classes)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()    
    
    train_size, val_size, test_size = size_list
    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = compute_steps_per_epoch(train_size)
    val_steps = compute_steps_per_epoch(val_size)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=5)
            
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # training
    trainer = Trainer(model, optimizer, loss_function)
    history = trainer.train(x_train, y_train, x_val=x_test, y_val=y_test, batch_size=32, epochs=10)
