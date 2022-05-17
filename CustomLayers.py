import tensorflow as tf

class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(ResidualLayer, self).__init__(name=name)
        # self.k=k
        super(ResidualLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        # config.update({"k": self.k})
        
        return config

    def call(self, inputs):
        return inputs[0] - inputs[1]

class AbsoluteResidual(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(AbsoluteResidual, self).__init__(name=name)
        # self.k=k
        super(AbsoluteResidual, self).__init__(**kwargs)

    def get_config(self):
        config = super(AbsoluteResidual, self).get_config()
        # config.update({"k": self.k})
        
        return config

    def call(self, inputs):
        return tf.keras.backend.abs(inputs[0] - inputs[1])