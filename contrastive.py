import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables


class ContrastiveModel(tf.keras.Model):
    def __init__(self, encoder, projector):
        super(ContrastiveModel, self).__init__()
        # self.classifier = classifier
        self.encoder = encoder
        self.projector = projector
        # self.c_steps = c_steps

    def compile(self, e_optimizer, e_loss_fn):
        super(ContrastiveModel, self).compile()
        # self.c_optimizer = c_optimizer
        self.e_optimizer = e_optimizer
        # self.c_loss_fn = c_loss_fn
        self.e_loss_fn = e_loss_fn
        # self.c_loss_metrics = tf.keras.metrics.Mean(name='c_loss')
        self.e_loss_metrics = tf.keras.metrics.Mean(name='e_loss')

    def summary(self):
        self.encoder.summary()
        self.projector.summary()
        # self.classifier.summary()

    @property
    def metrics(self):
        return [self.e_loss_metrics]

    def train_step(self, data):
        raw_ts, label = data
        # batch_size = tf.shape(raw_ts)[0]

        with tf.GradientTape() as tape:
            latent_codes = self.encoder(raw_ts, training = True)
            predictions = self.projector(latent_codes, training = True)
            e_loss = self.e_loss_fn(label, predictions)
        trainable_weights = self.encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(e_loss, trainable_weights)
        self.e_optimizer.apply_gradients(
            zip(grads, trainable_weights)
        )

        # for _ in range(self.c_steps):
        #     with tf.GradientTape() as tape:
        #         features = self.encoder(raw_ts, training = True)
        #         predictions = self.classifier(features, training = True)
        #         c_loss = self.c_loss_fn(label, predictions)
        #     grads = tape.gradient(c_loss, self.classifier.trainable_weights)
        #     self.c_optimizer.apply_gradients(
        #         zip(grads, self.classifier.trainable_weights)
        #     )

        # self.c_loss_metrics.update_state(c_loss)
        self.e_loss_metrics.update_state(e_loss)

        return {
            # "c_loss": self.c_loss_metrics.result(),
            "e_loss": self.e_loss_metrics.result(),
        }


class ContrastiveModel_ori(tf.keras.Model):
    def __init__(self, encoder, projector):
        super(ContrastiveModel_ori, self).__init__()
        # self.classifier = classifier
        self.encoder = encoder
        self.projector = projector
        # self.c_steps = c_steps

    def compile(self, e_optimizer, e_loss_fn):
        super(ContrastiveModel_ori, self).compile()
        # self.c_optimizer = c_optimizer
        self.e_optimizer = e_optimizer
        # self.c_loss_fn = c_loss_fn
        self.e_loss_fn = e_loss_fn
        # self.c_loss_metrics = tf.keras.metrics.Mean(name='c_loss')
        self.e_loss_metrics = tf.keras.metrics.Mean(name='e_loss')

    def summary(self):
        self.encoder.summary()
        self.projector.summary()
        # self.classifier.summary()

    @property
    def metrics(self):
        return [self.e_loss_metrics]

    def train_step(self, data):
        raw_ts, label = data
        # aug1 = raw_ts[:, 0, :, :]
        # aug2 = raw_ts[:, 1, :, :]
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

        # for _ in range(self.c_steps):
        #     with tf.GradientTape() as tape:
        #         features = self.encoder(raw_ts, training = True)
        #         predictions = self.classifier(features, training = True)
        #         c_loss = self.c_loss_fn(label, predictions)
        #     grads = tape.gradient(c_loss, self.classifier.trainable_weights)
        #     self.c_optimizer.apply_gradients(
        #         zip(grads, self.classifier.trainable_weights)
        #     )

        # self.c_loss_metrics.update_state(c_loss)
        self.e_loss_metrics.update_state(e_loss)

        return {
            # "c_loss": self.c_loss_metrics.result(),
            "e_loss": self.e_loss_metrics.result(),
        }
        