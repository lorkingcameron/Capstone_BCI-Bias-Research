from tf_keras.callbacks import EarlyStopping

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_epochs=10, patience=5, restore_best_weights=True, **kwargs):
        # Call the superclass constructor (EarlyStopping)
        super(DelayedEarlyStopping, self).__init__(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, **kwargs)
        self.min_epochs = min_epochs  # Minimum number of epochs before considering early stopping

    def on_epoch_end(self, epoch, logs=None):
        # Start monitoring early stopping only after min_epochs
        if epoch >= self.min_epochs:
            super(DelayedEarlyStopping, self).on_epoch_end(epoch, logs)
