from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models


class BasicBlock(layers.Layer):
    def __init__(self, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='SAME', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.stride = stride
        self.out_channels = out_channels
        self.downsample = layers.MaxPooling2D(pool_size=stride, strides=stride)
        self.project_conv = layers.Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False)
        self.project_bn = layers.BatchNormalization()

    def call(self, inputs, residual=None):
        if residual is None:
            residual = inputs
            if self.stride > 1:
                residual = self.downsample(residual)
            if self.out_channels != inputs.shape[3]:
                residual = self.project_conv(residual)
                residual = self.project_bn(residual)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = layers.add([x, residual])
        x = self.relu(x)

        return x


class Root(layers.Layer):
    def __init__(self, out_channels):
        super(Root, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, *inputs):
        x = tf.concat(inputs, axis=3)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Tree(layers.Layer):
    def __init__(self, levels, block, out_channels, stride=1, stage_root=False):
        super(Tree, self).__init__()

        self.levels = levels
        self.stage_root = stage_root

        if levels == 1:
            self.tree1 = block(out_channels, stride)
            self.tree2 = block(out_channels, 1)
        else:
            self.tree1 = Tree(levels - 1, block, out_channels, stride)
            self.tree2 = Tree(levels - 1, block, out_channels, 1)

        if levels == 1:
            self.root = Root(out_channels)

        self.downsample = None
        if stride > 1:
            self.downsample = layers.MaxPooling2D(pool_size=stride, strides=stride)

    def call(self, inputs, children=None):
        children = [] if children is None else children
        bottom = self.downsample(inputs) if self.downsample else inputs
        if self.stage_root:  # 如果是stage的root，需要把上一级stage的输出append进来
            children.append(bottom)
        x1 = self.tree1(inputs)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x1 = self.root(x1, x2, *children)
        else:
            children.append(x1)
            x1 = self.tree2(x1, children=children)
        return x1


# DLA
class DLA(keras.Model):
    def __init__(self, levels, channels, classes=1000, block=BasicBlock, return_levels=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.base_layer = keras.Sequential([
            layers.Conv2D(channels[0], kernel_size=7, strides=1, padding='SAME', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.stage0 = keras.Sequential([
            layers.Conv2D(channels[0], kernel_size=3, strides=1, padding='SAME', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.stage1 = keras.Sequential([
            layers.Conv2D(channels[1], kernel_size=3, strides=2, padding='SAME', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.stage2 = Tree(levels[2], block, channels[2], stride=2, stage_root=False)
        self.stage3 = Tree(levels[3], block, channels[3], stride=2, stage_root=True)
        self.stage4 = Tree(levels[4], block, channels[4], stride=2, stage_root=True)
        self.stage5 = Tree(levels[5], block, channels[5], stride=2, stage_root=True)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(classes, activation='softmax')

    def call(self, inputs):
        y = []
        x = self.base_layer(inputs)
        x = self.stage0(x)
        y.append(x)
        x = self.stage1(x)
        y.append(x)
        x = self.stage2(x)
        y.append(x)
        x = self.stage3(x)
        y.append(x)
        x = self.stage4(x)
        y.append(x)
        x = self.stage5(x)
        y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return x


def dla34(pretrained=None, **kwargs):
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla34')
    return model
