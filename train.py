import tensorflow as tf
from models.cnn import CNN
from dataloader import load_data
import os
import numpy as np
from tensorflow.python.client import device_lib
import math
from tensorflow.keras.utils import Progbar

# device 확인
# print(device_lib.list_local_devices())

# cpu/gpu 설정
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Trainer:

    def __init__(self, model, epochs, batch, loss_fn, optimizer):
        self.model = model 
        self.epochs = epochs 
        self.batch = batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        @tf.function
        def train_on_batch(self, x_batch_train, y_batch_train):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)    # 모델이 예측한 결과
                train_loss = self.loss_fn(y_batch_train, logits)     # 모델이 예측한 결과와 GT를 이용한 loss 계산

            grads = tape.gradient(train_loss, model.trainable_weights)  # gradient 계산
            self.optimizer.apply_gradients(zip(grads, model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청

            return train_loss, logits
        
    def train(self, train_dataset, train_metric, metrics_names = ['train_loss', 'train_acc', 'val_loss']):
        
        for epoch in range(self.epochs):
            print(f"Start of epoch {epoch}")
            
            # Progbar를 위해 스텝 수 만큼 dataset 저장
            train_dataset = train_dataset.take(steps_per_epoch)
            val_dataset = val_dataset.take(val_steps)
            
            progBar = Progbar(steps_per_epoch * self.batch, stateful_metrics=metrics_names)
            
            train_loss, val_loss = 100, 100
            
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # 컨텍스트(context) 안에서 실행된 모든 연산을 테이프(tape)에 기록
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    
                    # 모델의 예측결과와 label을 비교하여 loss_function 출력
                    loss_value = self.loss_fn(y_batch_train, logits)
                    # tf.print(loss_value)
                    
                # 기울기 계산
                grads = tape.gradient(loss_value, model.trainable_weights)
                
                # optimizer 적용
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                # 모델의 예측결과와 label로 accuracy 계산
                train_metric.update_state(y_batch_train, logits)
                
                # 5 step마다 loss값, 샘플 수, metric 출력
                if step % 5 ==0 :
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far : %d samples" % ((step+1)*self.batch))
                    print(train_metric.result().numpy)
                
                # 매 스텝마다 accuracy 출력
                train_acc = train_acc_metric.result()
                print(f"Training acc over epoch : {float(train_acc)}")
                
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
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()    
    
    train_size, val_size, test_size = size_list
    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = compute_steps_per_epoch(train_size)
    val_steps = compute_steps_per_epoch(val_size)
    
    print(steps_per_epoch)
    print(val_steps)

                
    # training
    trainer = Trainer(model=model, epochs=epoch, batch=batch_size, loss_fn=loss_function, optimizer=optimizer)
    trainer.train(train_dataset=train_ds, train_metric=train_acc_metric)