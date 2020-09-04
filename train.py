from datetime import datetime
from importlib import import_module
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import ExponentialMovingAverage

import argparse
import imagenet
import numpy as np
import os
import tensorflow as tf


class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(
            self,
            epochs,
            warmup_epochs,
            steps_per_epoch,
            base_lr,
            init_lr=0.0,
            initial_epoch=0):
        super(LearningRateScheduler, self).__init__()
        self.epochs = epochs
        self.last_batch = steps_per_epoch * initial_epoch
        self.epoch = 0
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.T_max = (epochs - warmup_epochs) * steps_per_epoch
        self.T_warmup = warmup_epochs * steps_per_epoch

    def on_batch_begin(self, batch, logs):
        self.last_batch += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.get_lr())

    def get_lr(self):
        if self.T_warmup == 0 or self.last_batch > self.T_warmup:
            curr_T = self.last_batch - self.T_warmup
            return 0.5 * self.base_lr * \
                (1 + np.cos(np.pi * curr_T / self.T_max))
        else:
            return self.init_lr + \
                (self.base_lr - self.init_lr) * self.last_batch / self.T_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048)
    parser.add_argument('--epochs', type=int, default=360)
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5)
    parser.add_argument('--base_lr', type=float, default=2.6)
    parser.add_argument('--init_lr', type=float, default=0.0)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument(
        '--imagenet_path',
        type=str,
        required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--save_every_epoch', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    args = parser.parse_args()
    print(args)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            exit()

    steps_per_epoch = 1281167 // args.batch_size
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.resume:
            model = tf.keras.models.load_model(args.resume)
        else:
            net = import_module(f'models.{args.model_name}')
            model = net.get_model()
    model.summary()
    train_dataset = imagenet.get_train_dataset(
        args.imagenet_path,
        args.batch_size,
        imagenet.NormalizeMethod.TF,
        use_color_jitter=True,
        use_one_hot=True,
        use_cache=args.use_cache,
        image_size=args.image_size).repeat().prefetch(10)
    val_dataset = imagenet.get_val_dataset(
        args.imagenet_path,
        args.batch_size,
        imagenet.NormalizeMethod.TF,
        use_one_hot=True,
        use_cache=args.use_cache,
        image_size=args.image_size)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join(args.checkpoint_path, f'{args.model_name}-{now_time}')
    os.makedirs(path, exist_ok=True)
    if args.save_every_epoch:
        saved_model_file = f'model.{{epoch:03d}}.h5'
    else:
        saved_model_file = 'model.best.h5'
    filepath = os.path.join(path, saved_model_file)
    callbacks = [
        LearningRateScheduler(
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            base_lr=args.base_lr,
            init_lr=args.init_lr,
            initial_epoch=args.initial_epoch),
        ModelCheckpoint(
            filepath,
            monitor='val_categorical_accuracy',
            verbose=0,
            save_best_only=(not args.save_every_epoch),
            save_weights_only=False,
            mode='auto',
            period=1),
        TensorBoard(f'{path}/logs')]
    if args.use_ema:
        callbacks.append(ExponentialMovingAverage())

    model.fit(train_dataset,
              epochs=args.epochs,
              steps_per_epoch=steps_per_epoch,
              shuffle=False,
              validation_data=val_dataset,
              callbacks=callbacks,
              initial_epoch=args.initial_epoch)

    model.save(f'{path}/model.h5')


if __name__ == '__main__':
    main()
