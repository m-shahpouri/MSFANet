import tensorflow as tf
@tf.keras.saving.register_keras_serializable(package='CustomLayers', name='channelwise_attention')
class CWA(tf.keras.layers.Layer):
    def __init__(self, backbone, nodes, block_name=1, in_channels=1, kernel_size=(1, 1), padding='same', dilation_rate=(1, 1), layer_activation='linear', last_activation='sigmoid', **kwargs):
        super(CWA, self).__init__(**kwargs)
        self.backbone = backbone
        self.nodes = nodes
        self.block_name = block_name
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.layer_activation = layer_activation
        self.last_activation = last_activation
        self.in_channels = in_channels

        self.globalAvgPool = tf.keras.layers.GlobalAveragePooling2D()
        self.attention = tf.keras.layers.Dense(units=self.nodes, activation=self.last_activation)

    def call(self, input):
        x = self.globalAvgPool(input)
        x = self.attention(x)
        return x

    def get_config(self):
        config = super(CWA, self).get_config()
        config.update(
            {
            'backbone': self.backbone,
            'nodes': self.nodes,
            'block_name' : self.block_name,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'dilation_rate' : self.dilation_rate,
            'last_activation' : self.last_activation
            })
        return config