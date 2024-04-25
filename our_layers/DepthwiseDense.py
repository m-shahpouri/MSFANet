import tensorflow as tf
@tf.keras.saving.register_keras_serializable(package='CustomLayers', name='depthwise_dense')
class DepthwiseDense(tf.keras.layers.Layer):
    def __init__(self, backbone, filters, block_name, in_channels, kernel_size=(1, 1), padding='same', dilation_rate=(1, 1), layer_activation='linear', last_activation='sigmoid', **kwargs):
        super(DepthwiseDense, self).__init__(**kwargs)
        self.backbone = backbone
        self.filters = filters
        self.block_name = block_name
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.layer_activation = layer_activation
        self.last_activation = last_activation
        self.in_channels = in_channels

        self.depthwise = tf.keras.layers.Conv2D(filters=self.in_channels, kernel_size=self.kernel_size, padding=self.padding,
                                        dilation_rate=self.dilation_rate, activation=self.layer_activation, groups=self.in_channels)
        # self.pointwise = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), activation=self.last_activation, padding=self.padding)
        self.pointwise = tf.keras.layers.Dense(units=self.filters, activation=self.last_activation)

    def call(self, input):
        x = self.depthwise(input)
        x = self.pointwise(x)
        return x

    def get_config(self):
        config = super(DepthwiseDense, self).get_config()
        config.update(
            {
            'backbone': self.backbone,
            'filters': self.filters,
            'block_name' : self.block_name,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'dilation_rate' : self.dilation_rate,
            'layer_activation' : self.layer_activation,
            'last_activation' : self.last_activation,
            'in_channels' : self.in_channels
            })
        return config