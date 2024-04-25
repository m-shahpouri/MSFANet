#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:33:47 2023

@author: rasa
"""


# Import packages
import tensorflow as tf
from our_layers.PWFE import PWFE
from our_layers.PSAE import PSAE

def create_model(model_name):
    # Clear all previously registered custom objects
    tf.keras.saving.get_custom_objects().clear()
    # Hyperparameters
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3

    model_input = tf.keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name='model_input')

    # # Backbone
    if model_name == 'vgg16':
        # vgg16
        preprocessed_vgg16_inputs = tf.keras.applications.vgg16.preprocess_input(model_input)
        vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=preprocessed_vgg16_inputs, input_shape=(HEIGHT, WIDTH, CHANNELS))
        # 2, 5, 9, 13, 17,
        # with preprocess layer 4, 7, 11, 15, 19
        # block_1_extractor = PWFE(backbone='vgg16', filters=vgg16.layers[4].filters,  block_name='block_1')(vgg16.layers[4].output)
        block_2_extractor = PWFE(backbone='vgg16', filters=vgg16.layers[7].filters,  in_channels=vgg16.layers[7].filters, block_name='block_2')(vgg16.layers[7].output)
        block_3_extractor = PWFE(backbone='vgg16', filters=vgg16.layers[11].filters, in_channels=vgg16.layers[11].filters,  block_name='block_3')(vgg16.layers[11].output)
        block_4_extractor = PWFE(backbone='vgg16', filters=vgg16.layers[15].filters, in_channels=vgg16.layers[15].filters, block_name='block_4')(vgg16.layers[15].output)
        block_5_extractor = PWFE(backbone='vgg16', filters=vgg16.layers[19].filters, in_channels=vgg16.layers[19].filters, block_name='block_5')(vgg16.layers[19].output)
        fusion_block = PSAE(backbone='vgg16', filters=4)([block_2_extractor, block_3_extractor, block_4_extractor, block_5_extractor])
        # fusion_block = PSAE(backbone='vgg16', filters=5)([block_1_extractor, block_2_extractor, block_3_extractor, block_4_extractor, block_5_extractor])
        # vgg16

    if model_name == 'resnet50':
        # ResNet50
        preprocessed_resnet50_input = tf.keras.applications.resnet.preprocess_input(model_input)
        resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, input_tensor=preprocessed_resnet50_input, input_shape=(HEIGHT, WIDTH, CHANNELS))
        filters = [256, 512, 1024, 2048]
        # 40, 82, 144, 176
        block_1_extractor = PWFE(backbone='resnet50', filters=filters[0], in_channels=filters[0], block_name='block_1')(resnet50.layers[40].output)
        block_2_extractor = PWFE(backbone='resnet50', filters=filters[1], in_channels=filters[1], block_name='block_2')(resnet50.layers[82].output)
        block_3_extractor = PWFE(backbone='resnet50', filters=filters[2], in_channels=filters[2], block_name='block_3')(resnet50.layers[144].output)
        block_4_extractor = PWFE(backbone='resnet50', filters=filters[3], in_channels=filters[3], block_name='block_4')(resnet50.layers[176].output)
        fusion_block = PSAE(backbone='resnet50', filters=4)([block_1_extractor, block_2_extractor, block_3_extractor, block_4_extractor])
        # ResNet50

    if model_name == 'resnet50_v2':
        # ResNet50_V2
        preprocessed_resnet50_v2_input = tf.keras.applications.resnet_v2.preprocess_input(model_input)
        resnet50_v2 = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, input_tensor=preprocessed_resnet50_v2_input, input_shape=(HEIGHT, WIDTH, CHANNELS))
        filters = [256, 512, 1024, 2048]
        # 41, 87, 155, 189
        block_1_extractor = PWFE(backbone='resnet50_v2', filters=filters[0], in_channels=filters[0], block_name='block_1')(resnet50_v2.layers[41].output)
        block_2_extractor = PWFE(backbone='resnet50_v2', filters=filters[1], in_channels=filters[1], block_name='block_2')(resnet50_v2.layers[87].output)
        block_3_extractor = PWFE(backbone='resnet50_v2', filters=filters[2], in_channels=filters[2], block_name='block_3')(resnet50_v2.layers[155].output)
        block_4_extractor = PWFE(backbone='resnet50_v2', filters=filters[3], in_channels=filters[3], block_name='block_4')(resnet50_v2.layers[189].output)
        fusion_block = PSAE(backbone='resnet50_v2', filters=4)([block_1_extractor, block_2_extractor, block_3_extractor, block_4_extractor])
        # ResNet50_V2

    if model_name == 'resnet50_rs':
        # # ResNet50_rs
        resnet50_rs = tf.keras.applications.resnet_rs.ResNetRS50(include_top=False, input_tensor=model_input, input_shape=(HEIGHT, WIDTH, CHANNELS))
        filters = [256, 512, 1024, 2048]
        # 64, 127, 221, 270
        block_1_extractor = PWFE(backbone='resnet50_rs', filters=filters[0], in_channels=filters[0], block_name='block_1')(resnet50_rs.layers[63].output)
        block_2_extractor = PWFE(backbone='resnet50_rs', filters=filters[1], in_channels=filters[1], block_name='block_2')(resnet50_rs.layers[127].output)
        block_3_extractor = PWFE(backbone='resnet50_rs', filters=filters[2], in_channels=filters[2], block_name='block_3')(resnet50_rs.layers[221].output)
        block_4_extractor = PWFE(backbone='resnet50_rs', filters=filters[3], in_channels=filters[3], block_name='block_4')(resnet50_rs.layers[270].output)
        fusion_block = PSAE(backbone='resnet50_rs', filters=4)([block_1_extractor, block_2_extractor, block_3_extractor, block_4_extractor])
        # ResNet50_rs


    if model_name == 'convnexttiny':
        # convnext
        convnext = tf.keras.applications.convnext.ConvNeXtTiny(include_top=False, input_tensor=model_input, model_name='convnext_tiny', input_shape=(HEIGHT, WIDTH, CHANNELS))
        filters = [96, 192, 384, 768]
        # 26 , 51, 124, 149
        block_1_extractor = PWFE(backbone='convnexttiny', filters=filters[0], in_channels=filters[0], block_name='block_1')(convnext.layers[26].output)
        block_2_extractor = PWFE(backbone='convnexttiny', filters=filters[1], in_channels=filters[1], block_name='block_2')(convnext.layers[51].output)
        block_3_extractor = PWFE(backbone='convnexttiny', filters=filters[2], in_channels=filters[2], block_name='block_3')(convnext.layers[124].output)
        block_4_extractor = PWFE(backbone='convnexttiny', filters=filters[3], in_channels=filters[3], block_name='block_4')(convnext.layers[149].output)
        fusion_block = PSAE(backbone='convnexttiny', filters=4)([block_1_extractor, block_2_extractor, block_3_extractor, block_4_extractor])
        # fusion_block = FusionBlock(backbone='convnexttiny', filters=4)([block_1_extractor, block_2_extractor, block_3_extractor, block_4_extractor])

        # convnext

    # Extractor
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, weight_decay=5e-3) # EC2Net 2023
    # optimizer = tf.keras.optimizers.AdamW() # 2017
    # optimizer = tf.keras.optimizers.Lion() # 2023

    # model = tf.keras.Model(inputs=convnext.input, outputs=fusion_block)
    model = tf.keras.Model(inputs=model_input, outputs=fusion_block)

    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae', f1score])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
    # model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model
