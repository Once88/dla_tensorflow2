import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
import numpy as np
import dla


class Identity(layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inp):
        return inp


class IDAUp(layers.Layer):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim

        # 从顶至底依次2个node,3个node,4个node
        # 设置各个节点对应的投影和上采样操作
        for i, c in enumerate(channels):
            # 若channel数与out_dim一致则无投影，否则投影到out_dim的channel数
            if c == out_dim:
                proj = Identity()
            else:
                proj = keras.Sequential([
                    layers.Conv2D(out_dim, kernel_size=1, strides=1, padding='SAME', use_bias=False),
                    layers.BatchNormalization(),
                    layers.ReLU()
                ])

            # 上采样因子，用于判断是否上采样，多少倍的上采样
            f = int(up_factors[i])

            # 若f为1则不做上采样，否则上采样到f倍
            if f == 1:
                up = Identity()
            else:
                up = layers.Conv2DTranspose(out_dim, f*2, strides=f, padding='SAME', output_padding=None, use_bias=False)
                # 初始化权重 todo
                # fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        # 设置node操作，即特征融合后将通道数减半的卷积层
        for i in range(1, len(channels)):
            node = keras.Sequential([
                layers.Conv2D(out_dim, kernel_size=node_kernel, strides=1, padding='SAME', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()
            ])
            setattr(self, 'node_' + str(i), node)

        # 初始化Conv2D和BN的参数 todo

    def call(self, nodes):
        # 检查输入数据维度是否合法
        assert len(self.channels) == len(nodes), '{} vs {} layers'.format(len(self.channels), len(nodes))
        nodes = list(nodes)
        for i, n in enumerate(nodes):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            nodes[i] = upsample(project(n))
        # 初始化x为最左侧的node
        x = nodes[0]
        # y用来返回给调用者以更新下次迭代传入的nodes
        y = []

        for i in range(1, len(nodes)):
            node = getattr(self, 'node_' + str(i))
            x = node(tf.concat([x, nodes[i]], 3))
            y.append(x)
        return x, y


class DLAUp(layers.Layer):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def call(self, in_layers):
        in_layers = list(in_layers)
        assert len(in_layers) > 1
        for i in range(len(in_layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(in_layers[-i - 2:])
            in_layers[-i - 1:] = y
        return x


class DLASeg(keras.Model):
    def __init__(self, base_name, classes, pretrained_base=None, down_ratio=2):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        # first_level是指从哪一个level开始进行上采样（IDA up），若down_ratio=2，则从level1开始
        self.first_level = int(np.log2(down_ratio))
        # 获取base_name对应的dla网络，相当于dla.dlaxxx()
        self.base = dla.__dict__[base_name](pretrained=pretrained_base,
                                            return_levels=True)
        channels = self.base.channels  # [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # [1, 2, 4, 8, 16] if first_level==1
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)  # [32, 64, 128, 256, 512], [1, 2, 4, 8, 16]
        self.fc = layers.Conv2D(classes, kernel_size=1, strides=1, use_bias=True)

        up_factor = 2 ** self.first_level
        if up_factor > 1:
            up = layers.Conv2DTranspose(classes, up_factor * 2, strides=up_factor, padding='SAME', output_padding=None, use_bias=False)
            # todo
            # fill_up_weights(up)
            # up.weight.requires_grad = False
        else:
            up = Identity()
        self.up = up

    def call(self, x):
        x = self.base(x)
        # 从first_level这一层开始做上采样
        x = self.dla_up(x[self.first_level:])  # if first_level=1, x.shape=[batch_size, 32, 112, 112]
        x = self.fc(x)  # if first_level=1, x.shape=[batch_size, classes, 112, 112]
        # y = tf.nn.log_softmax(self.up(x))  # if first_level=1, y.shape=[batch_size, classes, 224, 224]
        y = tf.nn.softmax(self.up(x))  # if first_level=1, y.shape=[batch_size, classes, 224, 224]
        # return y, x
        return y


def dla34up(classes, pretrained_base=None, **kwargs):
    model = DLASeg('dla34', classes, pretrained_base=pretrained_base, **kwargs)
    return model

