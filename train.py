import tensorflow as tf
from models.cnn import CNN
from dataloader import load_data
from tqdm import tqdm
import time

class Trainer:
    
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.loss_fn(y, logits)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss
    
    def train(self, train_ds, val_ds=None, epochs=10) :
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        progbar = tf.keras.utils.Progbar(train_size)
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            train_loss = tf.keras.metrics.Mean()
            train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
            
            for i, (x_batch, y_batch) in enumerate(train_ds):
                batch_loss = self.train_step(x_batch, y_batch)
                train_loss(batch_loss)
                train_acc(y_batch, self.model(x_batch))
                progbar.update(i)
                
            history['train_loss'].append(train_loss.result().numpy())
            history['train_acc'].append(train_acc.result().numpy())
            print(f"train_loss: {history['train_loss'][-1]:.4f} - train_acc: {history['train_acc'][-1]:.4f}")
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            
            if val_ds is not None:
                val_loss = tf.keras.metrics.Mean()
                val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
                
                for x_batch, y_batch in tqdm(val_ds):
                    val_loss(self.loss_fn(y_batch, self.model(x_batch)))
                    val_acc(y_batch, self.model(x_batch))
                    
                history['val_loss'].append(val_loss.result().numpy())
                history['val_acc'].append(val_acc.result().numpy())
                print(f"val_loss: {history['val_loss'][-1]:.4f} - val_acc: {history['val_acc'][-1]:.4f}")
                print("Time taken: %.2fs" % (time.time() - start_time))
        return history 
            
if __name__ == "__main__" :
    mnist = tf.keras.datasets.mnist
    train_ds, test_ds, val_ds, size_list =load_data(mnist)
    train_size, val_size, test_size = size_list
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    model = CNN(num_classes=10)
    epochs = 10
    batch_size = 32
    # Trainer 클래스를 사용하여 모델 학습
    trainer = Trainer(model, loss_fn=loss_function, optimizer=optimizer)
    history = trainer.train(train_ds, val_ds, epochs=10)
