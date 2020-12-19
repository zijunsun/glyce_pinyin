# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: glyce_cnn
@time: 2020/8/4 10:15

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.models.glyce.utils import channel_shuffle


class GlyphGroupCNN(nn.Module):
    """
    Glyce CNN Model
    Args:
        shuffle: if True, use shuffle channel
        num_features: output dim
        dropout: dropout
        groups: group_num of group_conv

    """

    def __init__(self, cnn_type='simple', font_channels=1, shuffle=False, ntokens=4000,
                 num_features=1024, final_width=2, dropout=0.5, groups=16):
        super(GlyphGroupCNN, self).__init__()
        self.aux_logits = False
        self.cnn_type = cnn_type
        output_channels = num_features
        self.conv1 = nn.Conv2d(font_channels, output_channels, 5)
        midchannels = output_channels // 4
        self.mid_groups = max(groups // 2, 1)
        self.downsample = nn.Conv2d(output_channels, midchannels, kernel_size=1, groups=self.mid_groups)
        self.max_pool = nn.AdaptiveMaxPool2d((final_width, final_width))
        self.num_features = num_features
        self.reweight_conv = nn.Conv2d(midchannels, output_channels, kernel_size=final_width, groups=groups)
        self.output_channels = output_channels
        self.shuffle = shuffle
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x):
        """
        encode a batch of chars into glyph features
        Args:
            x: input char ids: [batch, num_font, w, h]

        Returns:
            glyph features: [batch, self.num_features]
        """
        x = F.relu(self.conv1(x))  # [(seq_len*batchsize, Co, h, w), ...]*len(Ks)
        x = self.max_pool(x)  # n, c, 2, 2
        x = self.downsample(x)
        if self.shuffle:
            x = channel_shuffle(x, groups=2)
        x = F.relu(self.reweight_conv(x))
        if self.shuffle:
            x = channel_shuffle(x, groups=2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x  # (seq_len*batchsize, nfeats)

    def init_weights(self):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.uniform_(-initrange, initrange)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=1, std=0.001)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()


def run_model():
    """run models"""
    from backbones.models.glyce.utils import count_params
    conv = GlyphGroupCNN(num_features=768, groups=16, font_channels=8)
    print(conv)
    print('No. Parameters', count_params(conv))
    x = torch.rand([233, 8, 16, 16])
    y = conv(x)
    print(y.shape)


if __name__ == '__main__':
    run_model()
