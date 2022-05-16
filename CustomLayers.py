import tensorflow as tf

class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ResidualLayer, self).__init__()

    def call(self, inputs):
      return inputs[0] - inputs[1]

class AbsoluteResidual(tf.keras.layers.Layer):
    def __init__(self):
      super(AbsoluteResidual, self).__init__()

    def call(self, inputs):
      return tf.keras.backend.abs(inputs[0] - inputs[1])