from tensorflow.keras.callbacks import Callback
from tensorflow.python.framework import ops
from tensorflow.python.training import slot_creator

import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
import uuid

class ExponentialMovingAverage(Callback):
    def __init__(self, decay=0.999, filepath=f'/tmp/{uuid.uuid4()}', resume_ema=None, global_step=0):
        self.decay = decay
        self.global_step = global_step
        self.filepath = f'{filepath}.original.h5'
        self.ema_filepath = f'{filepath}.ema.h5'
        self.scope_name = 'ExponentialMovingAverage'
        self._averages = {}
        self.resume_ema = resume_ema
        super(ExponentialMovingAverage, self).__init__()

    def on_train_begin(self, logs={}):
        if self.resume_ema:
            model = tf.keras.models.load_model(self.resume_ema)
        else:
            model = self.model
        for weight in model.weights:
            with ops.init_scope():
                avg = slot_creator.create_slot(weight,
                    weight.initialized_value(),
                    self.scope_name,
                    colocate_with_primary=True)
            self._averages[weight.name] = avg

        self.model.save(self.filepath)

    def on_epoch_begin(self, epoch, logs={}):
        self.model.load_weights(self.filepath)

    def on_batch_end(self, batch, logs={}):
        self.average_update()

    def on_test_begin(self, logs={}):
        self.assign_shadow_weights()

    def average_update(self):
        # run in the end of each batch
        self.global_step += 1
        decay = min(self.decay, (1.0 + self.global_step) / (10.0 + self.global_step))
        for var in self.model.weights:
            self._averages[var.name] -=\
                (self._averages[var.name] - var) * (1.0 - decay)

    def assign_shadow_weights(self, backup=True):
        # run while you need to assign shadow weights (at end of each epoch or the total training)
        if backup:
            self.model.save(self.filepath)

        for weight in self.model.weights:
            tf.assign(weight, self._averages[weight.name])

        self.model.save(self.ema_filepath)
