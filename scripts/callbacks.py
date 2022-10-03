import numpy as np
import tensorflow as tf

class LinIncExpDecSchedule(tf.keras.callbacks.Callback):
  """Linear increase, exponential decrease schedule."""

  def __init__(
    self,
    half_life=10,
    min_lr=1e-5,
    max_lr=1e-4,
    ramp_up=20,
  ) -> None:
    super(RampSchedule, self).__init__()
    self.min_lr = min_lr
    self.delta = max_lr - min_lr
    self.ramp_up = ramp_up
    self.tau = half_life / tf.math.log(2.0)

  def on_epoch_begin(self, epoch, logs=None) -> None:
    if epoch <= self.ramp_up:
      lr = self.min_lr + self.delta * (epoch / self.ramp_up)
    else:
      time = (epoch - self.ramp_up)
      lr = self.min_lr + self.delta * tf.math.exp(-1.0 * time / self.tau)
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class EarlyStopWithMinEpochs(tf.keras.callbacks.Callback):
  """Early stopping with a minimum training period."""

  def __init__(
    self,
    min_epochs=20,
    patience=20,
    verbose=True
  ) -> None:
    self.best = np.Inf
    self.min_epochs = min_epochs
    self.patience = patience
    self.stopped_epoch = 0
    self.verbose = verbose
    self.wait = 0

  def on_epoch_end(self, epoch, logs=None) -> None:
    if epoch >= self.min_epochs:
      current = logs.get("val_loss")
      if current <= self.best:
        self.best = current
        self.wait = 0
        self.best_weights = self.model.get_weights()
      else:
        self.wait += 1
        if self.wait > self.patience:
          self.stopped_epoch = epoch
          self.model.stop_training = True
          self.model.set_weights(self.best_weights)
  
  def on_train_end(self, logs=None) -> None:
    if self.verbose & (self.stopped_epoch > 0):
      print(f"Early stopping triggered on epoch: {self.stopped_epoch}.")
