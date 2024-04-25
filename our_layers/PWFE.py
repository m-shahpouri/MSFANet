import tensorflow as tf
from .DepthwiseDense import DepthwiseDense
@tf.keras.saving.register_keras_serializable(package='CustomLayers', name='PWFEBlock')
class PWFE(tf.keras.layers.Layer):
    def __init__(self, backbone, filters, block_name, in_channels, kernel_size=(1, 1), padding='same', dilation_rate=(1, 1), layer_activation='linear', last_activation='sigmoid', *args, **kwargs):
        super(PWFE, self).__init__(*args, **kwargs)
        self.backbone = backbone
        self.filters = filters
        self.block_name = block_name
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.layer_activation = layer_activation
        self.last_activation = last_activation
        self.in_channels = in_channels

        # self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation=self.layer_activation)
        # self.conv_1 = tf.keras.layers.SeparableConv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding,
        #                                 dilation_rate=self.dilation_rate, activation=self.layer_activation)
        self.conv_1 = DepthwiseDense(filters=self.filters, in_channels=self.in_channels, kernel_size=self.kernel_size, padding=self.padding,
                                        dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='linear', backbone=self.backbone, block_name = self.block_name)

        self.layer_normalization = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_norm')

        self.relu_1 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_1')

        self.add = tf.keras.layers.Add(name=f'{self.block_name}_add')

        self.layer_normalization_2 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_norm_2')

        self.relu_2 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_2')

        # self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation=self.last_activation, name=f'{self.block_name}_conv_2')
        # self.conv_2 = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=self.kernel_size, padding=self.padding,
        #                                                 dilation_rate=self.dilation_rate, activation=self.last_activation)
        self.conv_2 = DepthwiseDense(filters=1, in_channels=self.filters, kernel_size=self.kernel_size, padding=self.padding,
                                        dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='sigmoid', backbone=self.backbone, block_name = self.block_name)

    def call(self, inputs, training=True):
        x = self.conv_1(inputs)
        x = self.layer_normalization(x)
        x = self.relu_1(x)
        x = self.add([x, inputs])
        x = self.layer_normalization_2(x)
        x = self.relu_2(x)
        x = self.conv_2(x)
        return x

    def get_config(self):
        config = super(PWFE, self).get_config()
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