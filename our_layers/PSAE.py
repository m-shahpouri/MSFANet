import tensorflow as tf
from .DepthwiseDense import DepthwiseDense
from .CWA import CWA
# @tf.keras.saving.register_keras_serializable(package='CustomLayers', name='PSAEBlock')
@tf.keras.saving.register_keras_serializable(package='CustomLayers', name='PSAEBlock')
class PSAE(tf.keras.layers.Layer):
    def __init__(self, backbone, filters, block_name='psae_block', kernel_size=(3, 3), padding='same',
                    dilation_rate=(1, 1), activation='linear', interpolation="nearest", *args, **kwargs):
        super(PSAE, self).__init__(name='psae', *args, **kwargs)
        self.backbone = backbone
        self.filters = filters
        self.block_name = block_name
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.interpolation = interpolation

        if self.backbone == 'vgg16':
            # vgg16
            self.upsample_2by2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_2by2')
            self.upsample_4by4 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_4by4')
            self.upsample_8by8 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_8by8')
            self.upsample_16by16 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_16by16')
            # vgg16

        if self.backbone == 'resnet50' or self.backbone == 'resnet50_rs' or self.backbone == 'convnexttiny':
            # resnet50, resnet50_rs, convnext
            self.upsample_4by4 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_4by4')
            self.upsample_8by8 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_8by8')
            self.upsample_16by16 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_16by16')
            self.upsample_32by32 = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_32by32')
            # resnet50, convnext

        if self.backbone == 'resnet50_v2':
            # resnet50_v2
            self.upsample_8by8 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_8by8')
            self.upsample_16by16 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_16by16')
            self.upsample_32by32_4 = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_32by32_4')
            self.upsample_32by32_5 = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation=self.interpolation,
                                                                name=f'{self.block_name}_upsample_32by32_5')
            # resnet50_v2

        self.concate = tf.keras.layers.Concatenate(name=f'{self.block_name}_concate')

        filters = [64, 128]

        # layer 1
        # self.conv_1_1 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_1_1')
        # self.conv_1_1 = tf.keras.layers.SeparableConv2D(filters=filters[0], kernel_size=(1, 1), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_1_1')
        self.conv_1_1 = DepthwiseDense(filters=filters[0], in_channels=4, kernel_size=(1, 1), padding=self.padding,
                                        dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='linear', backbone=self.backbone, block_name = self.block_name)

        self.layer_normalization_1_1 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_1_1')

        # self.relu_1_1 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_1_1')

        # self.conv_1_2 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_1_2')
        # self.conv_1_2 = tf.keras.layers.SeparableConv2D(filters=filters[0], kernel_size=(3, 3), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_1_2')
        self.conv_1_2 = DepthwiseDense(filters=filters[0], in_channels=4, kernel_size=(3, 3), padding=self.padding,
                                        dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='linear', backbone=self.backbone, block_name = self.block_name)

        self.layer_normalization_1_2 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_1_2')

        # self.relu_1_2 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_1_2')

        # self.conv_1_3 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(5, 5), padding=self.padding,
        #                                 dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_1_3')
        # self.conv_1_3 = tf.keras.layers.SeparableConv2D(filters=filters[0], kernel_size=(5, 5), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_1_3')
        self.conv_1_3 = DepthwiseDense(filters=filters[0], in_channels=4, kernel_size=(5, 5), padding=self.padding,
                                dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='linear', backbone=self.backbone, block_name = self.block_name)

        self.layer_normalization_1_3 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_1_3')

        # self.relu_1_3 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_1_3')

        # self.conv_1_4 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(7, 7), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation=self.activation, name=f'{self.block_name}_conv_1_4')
        # self.conv_1_4 = tf.keras.layers.SeparableConv2D(filters=filters[0], kernel_size=(7, 7), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation=self.activation, name=f'{self.block_name}_conv_1_4')
        self.conv_1_4 = DepthwiseDense(filters=filters[0], in_channels=4, kernel_size=(7, 7), padding=self.padding,
                                dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='linear', backbone=self.backbone, block_name = self.block_name)
        
        self.layer_normalization_1_4 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_1_4')

        # self.relu_1_4 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_1_4')

        # layer 2
        # self.conv_2_1 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_2_1')
        # self.conv_2_1 = tf.keras.layers.SeparableConv2D(filters=filters[1], kernel_size=(1, 1), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_2_1')
        self.conv_2_1 = DepthwiseDense(filters=filters[1], in_channels=filters[0], kernel_size=(1, 1), padding=self.padding,
                                dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='gelu', backbone=self.backbone, block_name = self.block_name)

        # self.layer_normalization_2_1 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_2_1')

        # self.relu_2_1 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_2_1')

        # self.conv_2_2 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(3, 3), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_2_2')
        # self.conv_2_2 = tf.keras.layers.SeparableConv2D(filters=filters[1], kernel_size=(3, 3), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_2_2')
        self.conv_2_2 = DepthwiseDense(filters=filters[1], in_channels=filters[0], kernel_size=(3, 3), padding=self.padding,
                                dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='gelu', backbone=self.backbone, block_name = self.block_name)

        # self.layer_normalization_2_2 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_2_2')

        # self.relu_2_2 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_2_2')

        # self.conv_2_3 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(5, 5), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_2_3')
        # self.conv_2_3 = tf.keras.layers.SeparableConv2D(filters=filters[1], kernel_size=(5, 5), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_2_3')
        self.conv_2_3 = DepthwiseDense(filters=filters[1], in_channels=filters[0], kernel_size=(5, 5), padding=self.padding,
                                dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='gelu', backbone=self.backbone, block_name = self.block_name)

        # self.layer_normalization_2_3 = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_2_3')

        # self.relu_2_3 = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_2_3')

        # self.conv_2_4 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(7, 7), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation=self.activation, name=f'{self.block_name}_conv_2_4')
        # self.conv_2_4 = tf.keras.layers.SeparableConv2D(filters=filters[1], kernel_size=(7, 7), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, activation=self.activation, name=f'{self.block_name}_conv_2_4')
        self.conv_2_4 = DepthwiseDense(filters=filters[1], in_channels=filters[0], kernel_size=(7, 7), padding=self.padding,
                                dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='gelu', backbone=self.backbone, block_name = self.block_name)

        self.parallel_add = tf.keras.layers.Add(name=f'{self.block_name}_parallel_add')
        

        # self.conv_concate = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), padding=self.padding,
        #                           dilation_rate=self.dilation_rate, activation='linear', name=f'{self.block_name}_conv_concate')
        # self.conv_concate = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=(1, 1), padding=self.padding,
        #                           dilation_rate=self.dilation_rate, activation='linear', name=f'{self.block_name}_separable_conv_concate')
        # self.conv_concate = DepthwiseDense(filters=4, in_channels=4, kernel_size=(1, 1), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='linear', backbone=self.backbone, block_name = self.block_name)
        self.channel_attention = CWA(backbone='backbone', nodes=128)

        # self.layer_normalization_concate = tf.keras.layers.LayerNormalization(name=f'{self.block_name}_layer_normalization_concate')

        # self.relu_concate = tf.keras.layers.ReLU(name=f'{self.block_name}_relu_concate')

        # self.conv_add = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), padding=self.padding,
        #                           dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_conv_add')
        # self.conv_add = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=(1, 1), padding=self.padding,
        #                           dilation_rate=self.dilation_rate, activation='relu', name=f'{self.block_name}_separable_conv_add')
        # self.conv_add = DepthwiseDense(filters=4, in_channels=filters[1], kernel_size=(1, 1), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='relu', backbone=self.backbone, block_name = self.block_name)

        self.multiply = tf.keras.layers.Multiply(name=f'{self.block_name}_multiply')

        self.conv_saliency_map = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding=self.padding,
                                        dilation_rate=self.dilation_rate, activation='sigmoid', name=f'{self.block_name}_conv_saliency_map')
        # self.conv_saliency_map = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=(1, 1), padding=self.padding,
        #                               dilation_rate=self.dilation_rate, activation='sigmoid', name=f'{self.block_name}_conv_saliency_map')
        # self.conv_saliency_map = DepthwiseDense(filters=4, in_channels=4, kernel_size=(1, 1), padding=self.padding,
        #                         dilation_rate=self.dilation_rate, layer_activation='linear', last_activation='sigmoid', backbone=self.backbone, block_name = self.block_name)

    def call(self, inputs):
        if self.backbone == 'vgg16':
            # VGG16
            # upsample_2by2 = self.upsample_2by2(inputs[1])
            # upsample_4by4 = self.upsample_4by4(inputs[2])
            # upsample_8by8 = self.upsample_8by8(inputs[3])
            # upsample_16by16 = self.upsample_16by16(inputs[4])
            upsample_2by2 = self.upsample_2by2(inputs[0])
            upsample_4by4 = self.upsample_4by4(inputs[1])
            upsample_8by8 = self.upsample_8by8(inputs[2])
            upsample_16by16 = self.upsample_16by16(inputs[3])
            concate = self.concate([upsample_2by2, upsample_4by4, upsample_8by8, upsample_16by16])
            # concate = self.concate([inputs[0], upsample_2by2, upsample_4by4, upsample_8by8, upsample_16by16])
            #

        if self.backbone == 'resnet50' or self.backbone == 'resnet50_rs' or self.backbone =='convnexttiny':
            # ResNet50, ResNet50_RS, ConvNextT
            upsample_4by4 = self.upsample_4by4(inputs[0])
            upsample_8by8 = self.upsample_8by8(inputs[1])
            upsample_16by16 = self.upsample_16by16(inputs[2])
            upsample_32by32 = self.upsample_32by32(inputs[3])
            concate = self.concate([upsample_4by4, upsample_8by8, upsample_16by16, upsample_32by32])
            #

        if self.backbone == 'resnet50_v2':
            # ResNet50_V2
            upsample_8by8 = self.upsample_8by8(inputs[0])
            upsample_16by16 = self.upsample_16by16(inputs[1])
            upsample_32by32_4 = self.upsample_32by32_4(inputs[2])
            upsample_32by32_5 = self.upsample_32by32_5(inputs[3])
            concate = self.concate([upsample_8by8, upsample_16by16, upsample_32by32_4, upsample_32by32_5])

        if self.backbone not in ['vgg16', 'resnet50', 'resnet50_v2', 'resnet50_rs', 'convnexttiny']:
            print(f'\n\nPllease insert correct backbone\n\n')
            quit()

        # layer 1
        x_1_1 = self.conv_1_1(concate)
        x_1_1 = self.layer_normalization_1_1(x_1_1)
        # x_1_1 = self.relu_1_1(x_1_1)

        x_1_2 = self.conv_1_2(concate)
        x_1_2 = self.layer_normalization_1_2(x_1_2)
        # x_1_2 = self.relu_1_2(x_1_2)

        x_1_3 = self.conv_1_3(concate)
        x_1_3 = self.layer_normalization_1_3(x_1_3)
        # x_1_3 = self.relu_1_3(x_1_3)

        x_1_4 = self.conv_1_4(concate)
        x_1_4 = self.layer_normalization_1_4(x_1_4)
        # x_1_4 = self.relu_1_4(x_1_4)

        # layer 2
        x_2_1 = self.conv_2_1(x_1_1)
        # x_2_1 = self.layer_normalization_2_1(x_2_1)
        # x_2_1 = self.relu_2_1(x_2_1)

        x_2_2 = self.conv_2_2(x_1_2)
        # x_2_2 = self.layer_normalization_2_2(x_2_2)
        # x_2_2 = self.relu_2_2(x_2_2)

        x_2_3 = self.conv_2_3(x_1_3)
        # x_2_3 = self.layer_normalization_2_3(x_2_3)
        # x_2_3 = self.relu_2_3(x_2_3)

        x_2_4 = self.conv_2_4(x_1_4)
        # x_2_4 = self.layer_normalization_2_4(x_2_4)
        # x_2_4 = self.relu_2_4(x_2_4)

        x = self.parallel_add([x_2_1, x_2_2, x_2_3, x_2_4])

        # skip_concate = self.conv_concate(concate)
        # skip_concate = self.layer_normalization_concate(skip_concate)
        # skip_concate = self.relu_concate(skip_concate)

        attention = self.channel_attention(concate)

        # x = self.layer_normalization(x)
        # x = self.relu(x)

        # x = self.conv_add(x)
        
        # x = self.add([x, concate])
        # x = self.multiply([x, skip_concate])
        x = self.multiply([x, attention])

        x = self.conv_saliency_map(x)
        return x

    def get_config(self):
        config = super(PSAE, self).get_config()
        config.update(
            {
            'backbone': self.backbone,
            'filters': self.filters,
            'block_name' : self.block_name,
            'kernel_size' : self.kernel_size,
            'padding' : self.padding,
            'dilation_rate' : self.dilation_rate,
            'activation' : self.activation,
            'interpolation' : self.interpolation
            })
        return config