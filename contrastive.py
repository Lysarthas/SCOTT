import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables


class ContrastiveModel(tf.keras.Model):
    def __init__(self, encoder, projector):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def compile(self, e_optimizer, e_loss_fn):
        super(ContrastiveModel, self).compile()
        self.e_optimizer = e_optimizer
        self.e_loss_fn = e_loss_fn
        self.e_loss_metrics = tf.keras.metrics.Mean(name='e_loss')

    def summary(self):
        self.encoder.summary()
        self.projector.summary()

    @property
    def metrics(self):
        return [self.e_loss_metrics]

    def train_step(self, data):
        raw_ts, label = data

        with tf.GradientTape() as tape:
            latent_codes = self.encoder(raw_ts, training = True)
            predictions = self.projector(latent_codes, training = True)
            e_loss = self.e_loss_fn(label, predictions)
        trainable_weights = self.encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(e_loss, trainable_weights)
        self.e_optimizer.apply_gradients(
            zip(grads, trainable_weights)
        )
        self.e_loss_metrics.update_state(e_loss)

        return {
            "e_loss": self.e_loss_metrics.result(),
        }


class ContrastiveModel_ori(tf.keras.Model):
    def __init__(self, encoder, projector):
        super(ContrastiveModel_ori, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def compile(self, e_optimizer, e_loss_fn):
        super(ContrastiveModel_ori, self).compile()
        self.e_optimizer = e_optimizer
        self.e_loss_fn = e_loss_fn
        self.e_loss_metrics = tf.keras.metrics.Mean(name='e_loss')

    def summary(self):
        self.encoder.summary()
        self.projector.summary()

    @property
    def metrics(self):
        return [self.e_loss_metrics]

    def train_step(self, data):
        raw_ts, label = data
        batch_size = tf.shape(raw_ts)[0]
        num_views = tf.shape(raw_ts)[1]
        ts_shape = tf.concat(([batch_size*num_views], tf.shape(raw_ts)[2:]), axis=0)
        raw_ts = tf.reshape(raw_ts, ts_shape)

        with tf.GradientTape() as tape:
            latent_codes = self.encoder(raw_ts, training = True)
            predictions = self.projector(latent_codes, training = True)
            _, f = predictions.shape
            predictions = tf.reshape(predictions, [batch_size, num_views, int(f)])
            e_loss = self.e_loss_fn(label, predictions)
        trainable_weights = self.encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(e_loss, trainable_weights)
        self.e_optimizer.apply_gradients(
            zip(grads, trainable_weights)
        )

        self.e_loss_metrics.update_state(e_loss)

        return {
            "e_loss": self.e_loss_metrics.result(),
        }
        