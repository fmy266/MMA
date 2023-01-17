import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 2
    def __init__(self, planes, width = 4, is_first_layer = False, is_last_layer = False):
        super(Block, self).__init__()
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.width = width

        self.rotue_conv_list = nn.ModuleList([nn.Conv2d(planes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False) for i in range(width)])
        self.rotue_bn_list = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(width)])
        self.rotue_shortcut_list = nn.ModuleList([nn.Sequential(
                                                  nn.Conv2d(planes, planes,
                                                  kernel_size=1, stride=1, bias=False),
                                                  nn.BatchNorm2d(planes))
                                                  for i in range(width//2)])

        self.downsample_conv_list = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(planes, planes * self.expansion, kernel_size=3,
                          stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(planes * self.expansion))   for i in range(width)
            ])

        if not self.is_last_layer:
            self.downsample_shortcut_list = nn.ModuleList([nn.Sequential(
                                                  nn.Conv2d(planes, planes * self.expansion, padding=1,
                                                  kernel_size=3, stride=2, bias=False),
                                                  nn.BatchNorm2d(planes * self.expansion))
                                                  for i in range(width-1)])


    def forward(self, x):

        out, downsample_conv_out, downsample_shortcut_out = [], [], []

        if self.is_first_layer:
            for i in range(0, self.width):
                if i == 0:
                    _out = F.relu(self.rotue_bn_list[i](self.rotue_conv_list[i](x)))
                    out.append(_out)
                else:
                    _out = F.relu(self.rotue_bn_list[i](self.rotue_conv_list[i](out[i-1])))
                    if i % 2 == 1:
                        if i == 1:
                            _out += self.rotue_shortcut_list[i//2](x)
                        else:
                            _out += self.rotue_shortcut_list[i//2](out[i-2])
                        _out = F.relu(_out)
                    out.append(_out)

            for i in range(self.width):
                downsample_conv_out.append(self.downsample_conv_list[i](out[i]))
                if i != self.width-1:
                    downsample_shortcut_out.append(self.downsample_shortcut_list[i](out[i]))

            return downsample_conv_out, downsample_shortcut_out

        elif self.is_last_layer:

            input_downsample_conv, input_shortcut_conv = x
            for i in range(0, self.width):
                if i == 0:
                    _out = F.relu(self.rotue_bn_list[i](self.rotue_conv_list[i](input_downsample_conv[i])))
                    out.append(_out)
                else:
                    _out = F.relu(self.rotue_bn_list[i](self.rotue_conv_list[i](out[i-1])))
                    _out = _out + input_downsample_conv[i]
                    _out = _out + input_shortcut_conv[i-1]
                    if i % 2 == 1:
                        if i == 1:
                            _out += self.rotue_shortcut_list[i//2](input_downsample_conv[0])
                        else:
                            _out += self.rotue_shortcut_list[i//2](out[i-2])
                        _out = F.relu(_out)
                    out.append(_out)

            for i in range(self.width):
                downsample_conv_out.append(self.downsample_conv_list[i](out[i]))

            return downsample_conv_out

        else:
            input_downsample_conv, input_shortcut_conv = x
            for i in range(0, self.width):
                if i == 0:
                    _out = F.relu(self.rotue_bn_list[i](self.rotue_conv_list[i](input_downsample_conv[i])))
                    out.append(_out)
                else:
                    _out = F.relu(self.rotue_bn_list[i](self.rotue_conv_list[i](out[i-1])))
                    _out = _out + input_downsample_conv[i]
                    _out = _out + input_shortcut_conv[i-1]
                    if i % 2 == 1:
                        if i == 1:
                            _out += self.rotue_shortcut_list[i//2](input_downsample_conv[0])
                        else:
                            _out += self.rotue_shortcut_list[i//2](out[i-2])
                        _out = F.relu(_out)
                    out.append(_out)

            for i in range(self.width):
                downsample_conv_out.append(self.downsample_conv_list[i](out[i]))
                if i != self.width-1:
                    downsample_shortcut_out.append(self.downsample_shortcut_list[i](out[i]))

            return downsample_conv_out, downsample_shortcut_out



class MultiTrackModel(nn.Module):
    '''
    Be careful to set the parameter expansion, where the solution of image
    must be suitable to it.
    For instance, to image with solution 3*32*32, the maximum number of blocks is 4.
    '''
    def __init__(self, channel = 3, num_classes = 10, init_filters = 32, expansion = Block.expansion, block_width = 4):
        super(MultiTrackModel, self).__init__()
        self.block_width = block_width
        # the default channel of image is set to 3.
        self.preprocessing_layer = nn.Sequential(nn.Conv2d(channel, init_filters, padding=1,
                                            kernel_size=3, stride=1, bias=False),
                                            nn.BatchNorm2d(init_filters))
        self.block1 = Block(init_filters, width = block_width, is_first_layer = True)
        self.block2 = Block(init_filters * expansion, width = block_width)
        self.block3 = Block(init_filters * expansion**2, width = block_width)
        self.block4 = Block(init_filters * expansion**3, width = block_width)
        self.block5 = Block(init_filters * expansion ** 4, width=block_width, is_last_layer=True)
        self.info = "{}x{}".format(2, block_width)

        # 4 is number of blocks
        self.linear_ls = nn.ModuleList([nn.Linear(init_filters * expansion ** 5, num_classes) for i in range(block_width)])


    def forward(self, x):
        '''
        output block_width * batch_size * num_classes
        '''
        out = self.preprocessing_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = torch.stack(out)
        out = out.mean(dim = [3, 4])
        out = [self.linear_ls[i](out[i]) for i in range(self.block_width)]
        return out
